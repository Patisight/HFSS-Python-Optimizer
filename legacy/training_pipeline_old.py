"""
训练管道 - 支持多样约束采样、持续学习和元学习
实现完整的训练流程，包括数据采样、模型训练、性能评估和模型管理

核心特性:
1. 多样约束采样: Latin Hypercube、分层采样、自适应采样
2. 持续学习: EWC (Elastic Weight Consolidation) 防止灾难性遗忘
3. 元学习支持: MAML框架的基础实现
4. 训练监控: 实时性能跟踪和可视化
5. 模型管理: 自动保存、版本控制和性能比较
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import pickle
import warnings

# 导入相关模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.parameterized_pixel_env import ParameterizedPixelAntennaEnv, ConstraintConfig
from constraint.constraint_sampler import ConstraintSampler, SamplingStrategy
from constraint.constraint_manager import ConstraintManager
from agent.generalized_ppo_agent import GeneralizedPPOAgent, AgentConfig
from constraint.constraint_manager import ConstraintGroup

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    total_timesteps: int = 100000
    eval_interval: int = 10000
    save_interval: int = 20000
    constraint_schedule: str = 'sequential'  # 'sequential', 'random', 'curriculum'

    # 约束采样
    n_training_constraints: int = 50
    n_eval_constraints: int = 20
    constraint_sampling_strategy: str = 'latin_hypercube'
    # 多频段约束组支持
    use_constraint_groups: bool = False
    n_training_groups: int = 30
    n_eval_groups: int = 10
    bands_per_group_min: int = 2
    bands_per_group_max: int = 3
    enforce_non_overlap: bool = True
    
    # 持续学习
    enable_continual_learning: bool = True
    ewc_lambda: float = 0.5  # EWC正则化强度
    memory_buffer_size: int = 1000  # 经验回放缓冲区大小
    
    # 元学习
    enable_meta_learning: bool = False
    meta_batch_size: int = 4
    meta_lr: float = 1e-3
    inner_lr: float = 1e-2
    inner_steps: int = 5
    
    # 课程学习
    enable_curriculum: bool = True
    curriculum_strategy: str = 'difficulty'  # 'difficulty', 'frequency', 'mixed'
    
    # 性能监控
    enable_wandb: bool = False
    wandb_project: str = 'pixel_antenna_rl'
    log_level: str = 'INFO'
    
    # 模型管理
    model_save_dir: str = 'models'
    keep_best_n_models: int = 5
    performance_metric: str = 'success_rate'  # 'success_rate', 'mean_reward'

class EWCRegularizer:
    """
    Elastic Weight Consolidation (EWC) 正则化器
    防止在学习新任务时遗忘旧任务
    """
    
    def __init__(self, model: nn.Module, lambda_reg: float = 0.5):
        self.model = model
        self.lambda_reg = lambda_reg
        self.fisher_information = {}
        self.optimal_params = {}
        
    def compute_fisher_information(self, dataloader, device='cpu'):
        """计算Fisher信息矩阵"""
        self.model.eval()
        fisher_info = {}
        
        # 初始化Fisher信息
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param.data)
                
        # 计算Fisher信息
        n_samples = 0
        for batch in dataloader:
            self.model.zero_grad()
            
            # 前向传播
            if isinstance(batch, dict):
                outputs = self.model(**batch)
            else:
                outputs = self.model(batch)
                
            # 计算损失（使用对数似然）
            if hasattr(outputs, 'logits'):
                loss = torch.nn.functional.log_softmax(outputs.logits, dim=-1).sum()
            else:
                loss = outputs.sum()
                
            # 反向传播
            loss.backward()
            
            # 累积梯度平方
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
                    
            n_samples += 1
            
        # 平均化Fisher信息
        for name in fisher_info:
            fisher_info[name] /= n_samples
            
        self.fisher_information = fisher_info
        
        # 保存当前最优参数
        self.optimal_params = {
            name: param.data.clone() 
            for name, param in self.model.named_parameters() 
            if param.requires_grad
        }
        
    def compute_ewc_loss(self) -> torch.Tensor:
        """计算EWC损失"""
        if not self.fisher_information:
            return torch.tensor(0.0)
            
        ewc_loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                ewc_loss += (fisher * (param - optimal) ** 2).sum()
                
        return self.lambda_reg * ewc_loss

class MetaLearner:
    """
    元学习器 - 基于MAML的实现
    学习如何快速适应新约束
    """
    
    def __init__(self, 
                 model: nn.Module,
                 meta_lr: float = 1e-3,
                 inner_lr: float = 1e-2,
                 inner_steps: int = 5):
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        
        # 元优化器
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        
    def meta_update(self, task_batch: List[Dict]) -> Dict[str, float]:
        """执行元更新"""
        meta_loss = 0.0
        meta_metrics = defaultdict(list)
        
        for task_data in task_batch:
            # 内循环：快速适应
            adapted_params = self._inner_loop(task_data['support'])
            
            # 外循环：在查询集上评估
            query_loss, query_metrics = self._evaluate_on_query(
                task_data['query'], adapted_params
            )
            
            meta_loss += query_loss
            for key, value in query_metrics.items():
                meta_metrics[key].append(value)
                
        # 元梯度更新
        meta_loss /= len(task_batch)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        # 平均化指标
        avg_metrics = {key: np.mean(values) for key, values in meta_metrics.items()}
        avg_metrics['meta_loss'] = meta_loss.item()
        
        return avg_metrics
        
    def _inner_loop(self, support_data: Dict) -> Dict[str, torch.Tensor]:
        """内循环：快速适应"""
        # 创建参数副本
        adapted_params = {
            name: param.clone() 
            for name, param in self.model.named_parameters()
        }
        
        # 内循环更新
        for step in range(self.inner_steps):
            # 计算支持集损失
            loss = self._compute_task_loss(support_data, adapted_params)
            
            # 计算梯度
            grads = torch.autograd.grad(
                loss, adapted_params.values(), 
                create_graph=True, retain_graph=True
            )
            
            # 更新参数
            for (name, param), grad in zip(adapted_params.items(), grads):
                adapted_params[name] = param - self.inner_lr * grad
                
        return adapted_params
        
    def _evaluate_on_query(self, 
                          query_data: Dict, 
                          adapted_params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """在查询集上评估"""
        query_loss = self._compute_task_loss(query_data, adapted_params)
        
        # 计算其他指标
        with torch.no_grad():
            predictions = self._forward_with_params(query_data['inputs'], adapted_params)
            accuracy = self._compute_accuracy(predictions, query_data['targets'])
            
        metrics = {'query_accuracy': accuracy}
        return query_loss, metrics
        
    def _compute_task_loss(self, task_data: Dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算任务损失"""
        predictions = self._forward_with_params(task_data['inputs'], params)
        loss = nn.functional.mse_loss(predictions, task_data['targets'])
        return loss
        
    def _forward_with_params(self, inputs: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """使用给定参数进行前向传播"""
        # 这里需要根据具体模型结构实现
        # 简化实现，实际需要根据模型架构调整
        return self.model(inputs)  # 简化版本
        
    def _compute_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """计算准确率"""
        # 根据具体任务定义准确率
        mse = nn.functional.mse_loss(predictions, targets)
        return float(1.0 / (1.0 + mse.item()))  # 简化的准确率定义

class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = defaultdict(deque)
        self.constraint_performance = defaultdict(dict)
        
    def update(self, metrics: Dict[str, float], constraint_id: Optional[str] = None):
        """更新性能指标"""
        timestamp = time.time()
        
        for metric_name, value in metrics.items():
            self.metrics_history[metric_name].append((timestamp, value))
            
            # 保持窗口大小
            if len(self.metrics_history[metric_name]) > self.window_size:
                self.metrics_history[metric_name].popleft()
                
        # 记录约束特定性能
        if constraint_id:
            if constraint_id not in self.constraint_performance:
                self.constraint_performance[constraint_id] = defaultdict(list)
                
            for metric_name, value in metrics.items():
                self.constraint_performance[constraint_id][metric_name].append(value)
                
    def get_recent_average(self, metric_name: str, n_recent: int = 10) -> float:
        """获取最近N个值的平均"""
        if metric_name not in self.metrics_history:
            return 0.0
            
        recent_values = list(self.metrics_history[metric_name])[-n_recent:]
        if not recent_values:
            return 0.0
            
        return np.mean([value for _, value in recent_values])
        
    def get_constraint_performance(self, constraint_id: str) -> Dict[str, float]:
        """获取特定约束的性能统计"""
        if constraint_id not in self.constraint_performance:
            return {}
            
        stats = {}
        for metric_name, values in self.constraint_performance[constraint_id].items():
            if values:
                stats[f"{metric_name}_mean"] = np.mean(values)
                stats[f"{metric_name}_std"] = np.std(values)
                stats[f"{metric_name}_latest"] = values[-1]
                
        return stats
        
    def plot_metrics(self, save_path: Optional[str] = None):
        """绘制性能指标"""
        if not self.metrics_history:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (metric_name, history) in enumerate(self.metrics_history.items()):
            if i >= len(axes):
                break
                
            timestamps, values = zip(*history) if history else ([], [])
            axes[i].plot(timestamps, values)
            axes[i].set_title(f'{metric_name}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Value')
            axes[i].grid(True)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

class TrainingPipeline:
    """
    训练管道主类
    
    协调所有训练组件，实现完整的训练流程
    """
    
    def __init__(self, 
                 env: ParameterizedPixelAntennaEnv,
                 config: TrainingConfig,
                 agent_config: Optional[AgentConfig] = None):
        """
        初始化训练管道
        
        Args:
            env: 参数化环境
            config: 训练配置
            agent_config: 智能体配置
        """
        self.env = env
        self.config = config
        self.agent_config = agent_config or AgentConfig()
        
        # 创建保存目录
        self.save_dir = Path(config.model_save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.constraint_sampler = ConstraintSampler()
        self.constraint_manager = ConstraintManager()
        self.performance_tracker = PerformanceTracker()
        
        # 初始化智能体
        self.agent = GeneralizedPPOAgent(
            env=env,
            config=agent_config,
            model_save_path=str(self.save_dir / 'best_model')
        )
        
        # 持续学习组件
        if config.enable_continual_learning:
            self.ewc_regularizer = EWCRegularizer(
                self.agent.model.policy, 
                lambda_reg=config.ewc_lambda
            )
        else:
            self.ewc_regularizer = None
            
        # 元学习组件
        if config.enable_meta_learning:
            self.meta_learner = MetaLearner(
                self.agent.model.policy,
                meta_lr=config.meta_lr,
                inner_lr=config.inner_lr,
                inner_steps=config.inner_steps
            )
        else:
            self.meta_learner = None
            
        # 日志配置
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # 训练状态
        self.training_state = {
            'current_step': 0,
            'best_performance': -float('inf'),
            'training_constraints': [],
            'eval_constraints': [],
            'model_versions': []
        }
        
    def run_training(self) -> Dict[str, Any]:
        """运行完整训练流程"""
        self.logger.info("开始训练流程")
        start_time = time.time()
        
        try:
            # 1. 准备约束数据
            self._prepare_constraints()
            
            # 2. 基础训练阶段
            self._run_base_training()
            
            # 3. 泛化训练阶段
            if self.config.enable_continual_learning or self.config.enable_meta_learning:
                self._run_generalization_training()
                
            # 4. 最终评估
            final_results = self._run_final_evaluation()
            
            # 5. 保存结果
            self._save_training_results(final_results)
            
            training_time = time.time() - start_time
            self.logger.info(f"训练完成，总耗时: {training_time:.2f}秒")
            
            return {
                'training_time': training_time,
                'final_results': final_results,
                'training_state': self.training_state
            }
            
        except Exception as e:
            self.logger.error(f"训练过程中出现错误: {str(e)}")
            raise
            
    def _prepare_constraints(self):
        """准备训练和评估约束"""
        self.logger.info("准备约束数据")
        if self.config.constraint_sampling_strategy == 'latin_hypercube':
            strategy = SamplingStrategy.LATIN_HYPERCUBE
        elif self.config.constraint_sampling_strategy == 'stratified':
            strategy = SamplingStrategy.STRATIFIED
        else:
            strategy = SamplingStrategy.RANDOM
        
        if self.config.use_constraint_groups:
            # 生成训练约束组
            self.training_state['training_constraints'] = self.constraint_sampler.sample_constraint_groups(
                n_groups=self.config.n_training_groups,
                bands_per_group=(self.config.bands_per_group_min, self.config.bands_per_group_max),
                strategy_override=strategy,
                ensure_non_overlap=self.config.enforce_non_overlap
            )
            # 生成评估约束组
            np.random.seed(42)
            self.training_state['eval_constraints'] = self.constraint_sampler.sample_constraint_groups(
                n_groups=self.config.n_eval_groups,
                bands_per_group=(self.config.bands_per_group_min, self.config.bands_per_group_max),
                strategy_override=SamplingStrategy.RANDOM,
                ensure_non_overlap=self.config.enforce_non_overlap
            )
            np.random.seed()
            self.logger.info(f"生成训练约束组: {len(self.training_state['training_constraints'])}个")
            self.logger.info(f"生成评估约束组: {len(self.training_state['eval_constraints'])}个")
        else:
            # 生成单约束
            self.training_state['training_constraints'] = self.constraint_sampler.sample_constraints(
                n_samples=self.config.n_training_constraints,
                strategy_override=strategy
            )
            np.random.seed(42)
            self.training_state['eval_constraints'] = self.constraint_sampler.sample_constraints(
                n_samples=self.config.n_eval_constraints,
                strategy_override=SamplingStrategy.RANDOM
            )
            np.random.seed()
            self.logger.info(f"生成训练约束: {len(self.training_state['training_constraints'])}个")
            self.logger.info(f"生成评估约束: {len(self.training_state['eval_constraints'])}个")
        
        # 将采样的约束落盘，方便查看
        try:
            self._save_constraints_snapshot(
                self.training_state['training_constraints'],
                Path(self.config.model_save_dir) / 'sampled_training_constraints.json',
                use_groups=self.config.use_constraint_groups
            )
            self._save_constraints_snapshot(
                self.training_state['eval_constraints'],
                Path(self.config.model_save_dir) / 'sampled_eval_constraints.json',
                use_groups=self.config.use_constraint_groups
            )
            self.logger.info("已保存采样约束到 models 目录")
        except Exception as e:
            self.logger.warning(f"采样约束保存失败: {e}")
    def _save_constraints_snapshot(self, constraints, filepath: Path, use_groups: bool = False):
        """保存采样约束到JSON，支持约束组或单约束"""
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            if use_groups:
                payload = {
                    'use_groups': True,
                    'groups': [
                        {
                            'name': g.name,
                            'bands': [asdict(c) for c in g.constraints]
                        } for g in constraints
                    ]
                }
            else:
                payload = {
                    'use_groups': False,
                    'constraints': [asdict(c) for c in constraints]
                }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    def _run_base_training(self):
        """运行基础训练"""
        self.logger.info("开始基础训练阶段")
        
        # 准备约束调度
        training_constraints = self.training_state['training_constraints']
        schedule = self.config.constraint_schedule
        if schedule == 'curriculum' and self.config.enable_curriculum:
            training_constraints = self._apply_curriculum_learning(training_constraints)
            schedule = 'curriculum'
            
        # 训练智能体
        training_stats = self.agent.train(
            total_timesteps=self.config.total_timesteps,
            constraints=training_constraints,
            constraint_schedule=schedule,
            save_interval=self.config.save_interval,
            callback_kwargs={
                'log_interval': 200,
                'adaptation_interval': 2000,
                'verbose': 1,
            }
        )
        
        # 更新训练状态
        self.training_state['current_step'] = self.config.total_timesteps
        self.training_state.update(training_stats)
        
        # 定期评估
        if self.config.eval_interval > 0:
            self._run_periodic_evaluation()
            
    def _run_generalization_training(self):
        """运行泛化训练"""
        self.logger.info("开始泛化训练阶段")
        
        if self.config.enable_meta_learning and self.meta_learner:
            self._run_meta_learning()
            
        if self.config.enable_continual_learning and self.ewc_regularizer:
            self._run_continual_learning()
            
    def _run_meta_learning(self):
        """运行元学习训练"""
        self.logger.info("执行元学习训练")
        
        # 准备元学习任务批次
        n_meta_batches = len(self.training_state['training_constraints']) // self.config.meta_batch_size
        
        for batch_idx in range(n_meta_batches):
            # 构建任务批次
            task_batch = self._create_meta_task_batch(batch_idx)
            
            # 执行元更新
            meta_metrics = self.meta_learner.meta_update(task_batch)
            
            # 记录性能
            self.performance_tracker.update(meta_metrics, f"meta_batch_{batch_idx}")
            
            if batch_idx % 10 == 0:
                self.logger.info(f"元学习批次 {batch_idx}/{n_meta_batches}, 损失: {meta_metrics['meta_loss']:.4f}")
                
    def _run_continual_learning(self):
        """运行持续学习"""
        self.logger.info("执行持续学习训练")
        if self.config.use_constraint_groups:
            new_constraints = self.constraint_sampler.sample_constraint_groups(
                n_groups=5,
                bands_per_group=(self.config.bands_per_group_min, self.config.bands_per_group_max),
                strategy_override=SamplingStrategy.RANDOM,
                ensure_non_overlap=self.config.enforce_non_overlap
            )
        else:
            new_constraints = self.constraint_sampler.sample_constraints(
                n_samples=10,
                strategy_override=SamplingStrategy.RANDOM
            )
        
        fine_tune_stats = self.agent.fine_tune(
            new_constraints=new_constraints,
            fine_tune_steps=5000,
            learning_rate_factor=0.1
        )
        self.logger.info(f"持续学习完成，平均改进: {np.mean([s['improvement'] for s in fine_tune_stats.values()]):.4f}")
        
    def _run_periodic_evaluation(self):
        """运行定期评估"""
        eval_results = self.agent.evaluate(
            constraints=self.training_state['eval_constraints'],
            n_episodes_per_constraint=5
        )
        
        # 更新性能跟踪
        overall_performance = eval_results['overall_success_rate']
        self.performance_tracker.update({
            'eval_success_rate': overall_performance,
            'eval_mean_reward': eval_results['overall_mean_reward']
        })
        
        # 检查是否是最佳性能
        if overall_performance > self.training_state['best_performance']:
            self.training_state['best_performance'] = overall_performance
            self._save_best_model()
            
        self.logger.info(f"评估完成，成功率: {overall_performance:.2%}")
        
    def _run_final_evaluation(self) -> Dict[str, Any]:
        """运行最终评估"""
        self.logger.info("开始最终评估")
        
        # 在所有评估约束上进行详细评估
        final_results = self.agent.evaluate(
            constraints=self.training_state['eval_constraints'],
            n_episodes_per_constraint=20
        )
        
        # 泛化能力测试：生成新的测试约束或约束组
        if self.config.use_constraint_groups:
            generalization_constraints = self.constraint_sampler.sample_constraint_groups(
                n_groups=5,
                bands_per_group=(self.config.bands_per_group_min, self.config.bands_per_group_max),
                strategy_override=SamplingStrategy.RANDOM,
                ensure_non_overlap=self.config.enforce_non_overlap
            )
        else:
            generalization_constraints = self.constraint_sampler.sample_constraints(
                n_samples=10,
                strategy_override=SamplingStrategy.RANDOM
            )
        
        generalization_results = self.agent.evaluate(
            constraints=generalization_constraints,
            n_episodes_per_constraint=10
        )
        
        # 合并结果
        final_results['generalization_results'] = generalization_results
        final_results['generalization_success_rate'] = generalization_results['overall_success_rate']
        
        return final_results
        
    def _apply_curriculum_learning(self, constraints: List) -> List:
        """应用课程学习（兼容约束组）"""
        if self.config.curriculum_strategy == 'difficulty':
            # 按难度排序（简单到复杂）
            def difficulty_score(c) -> float:
                if isinstance(c, ConstraintGroup):
                    if len(c.constraints) == 0:
                        return 0.0
                    scores = []
                    for sc in c.constraints:
                        bandwidth = sc.freq_high - sc.freq_low
                        target_strictness = abs(sc.target_s11)
                        tolerance_strictness = 1.0 / max(sc.tolerance, 1e-3)
                        scores.append(bandwidth / 1e9 - target_strictness * 0.1 - tolerance_strictness * 0.5)
                    return float(np.mean(scores))
                else:
                    bandwidth = c.freq_high - c.freq_low
                    target_strictness = abs(c.target_s11)
                    tolerance_strictness = 1.0 / max(c.tolerance, 1e-3)
                    return bandwidth / 1e9 - target_strictness * 0.1 - tolerance_strictness * 0.5
            return sorted(constraints, key=difficulty_score, reverse=True)
            
        elif self.config.curriculum_strategy == 'frequency':
            # 按频率范围排序（约束组按最小频率排序）
            def freq_key(c):
                if isinstance(c, ConstraintGroup):
                    if len(c.constraints) == 0:
                        return 0.0
                    return min(sc.freq_low for sc in c.constraints)
                else:
                    return c.freq_low
            return sorted(constraints, key=freq_key)
        else:
            return constraints
            

            
    def _create_meta_task_batch(self, batch_idx: int) -> List[Dict]:
        """创建元学习任务批次，若约束为约束组则跳过"""
        task_batch = []
        start_idx = batch_idx * self.config.meta_batch_size
        end_idx = start_idx + self.config.meta_batch_size
        batch_constraints = self.training_state['training_constraints'][start_idx:end_idx]
        for constraint in batch_constraints:
            if isinstance(constraint, ConstraintGroup):
                continue
            task_data = {
                'constraint': constraint,
                'support': self._generate_task_data(constraint, n_samples=10),
                'query': self._generate_task_data(constraint, n_samples=5)
            }
            task_batch.append(task_data)
        return task_batch
        
    def _generate_task_data(self, constraint: ConstraintConfig, n_samples: int) -> Dict:
        """为特定约束生成任务数据"""
        # 简化实现，实际需要从环境中收集数据
        inputs = torch.randn(n_samples, self.env.observation_space.shape[0])
        targets = torch.randn(n_samples, self.env.action_space.shape[0])
        
        return {
            'inputs': inputs,
            'targets': targets,
            'constraint': constraint
        }
        
    def _save_best_model(self):
        """保存最佳模型"""
        model_path = self.save_dir / f"best_model_step_{self.training_state['current_step']}"
        self.agent.save_model(str(model_path))
        
        # 记录模型版本
        self.training_state['model_versions'].append({
            'path': str(model_path),
            'step': self.training_state['current_step'],
            'performance': self.training_state['best_performance'],
            'timestamp': time.time()
        })
        
        # 保持最佳N个模型
        if len(self.training_state['model_versions']) > self.config.keep_best_n_models:
            # 删除最旧的模型
            oldest_model = self.training_state['model_versions'].pop(0)
            try:
                Path(oldest_model['path']).unlink(missing_ok=True)
                Path(f"{oldest_model['path']}_config.json").unlink(missing_ok=True)
            except:
                pass
                
    def _save_training_results(self, final_results: Dict[str, Any]):
        """保存训练结果"""
        results_path = self.save_dir / 'training_results.json'
        
        # 准备保存数据
        save_data = {
            'config': asdict(self.config),
            'agent_config': asdict(self.agent_config),
            'training_state': self.training_state,
            'final_results': final_results,
            'timestamp': time.time()
        }
        
        # 保存JSON
        with open(results_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
            
        # 保存性能图表
        plot_path = self.save_dir / 'performance_plots.png'
        self.performance_tracker.plot_metrics(str(plot_path))
        
        self.logger.info(f"训练结果已保存到: {results_path}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint_path = Path(checkpoint_path)
        
        # 加载模型
        model_path = checkpoint_path / 'best_model'
        if model_path.exists():
            self.agent.load_model(str(model_path))
            
        # 加载训练状态
        state_path = checkpoint_path / 'training_results.json'
        if state_path.exists():
            with open(state_path, 'r') as f:
                data = json.load(f)
                self.training_state = data.get('training_state', {})
                
        self.logger.info(f"检查点已加载: {checkpoint_path}")
        
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        return {
            'config': asdict(self.config),
            'training_state': self.training_state,
            'performance_summary': {
                'best_performance': self.training_state['best_performance'],
                'total_steps': self.training_state['current_step'],
                'n_model_versions': len(self.training_state['model_versions'])
            },
            'agent_info': self.agent.get_model_info()
        }