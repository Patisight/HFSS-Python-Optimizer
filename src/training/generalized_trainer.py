"""
泛化训练器

整合环境、智能体、约束管理和课程学习，实现完整的参数化强化学习训练流程
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict, deque
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ..environment.parameterized_env import ParameterizedPixelAntennaEnv
from ..agent.generalized_agent import GeneralizedPPOAgent
from ..agent.agent_config import AgentConfig
from ..config.constraint_config import ConstraintConfig, ConstraintManager
from ..config.constraint_sampler import ConstraintSampler, SamplingStrategy, SamplingConfig
from .training_config import TrainingConfig
from .curriculum_scheduler import CurriculumScheduler

logger = logging.getLogger(__name__)

class GeneralizedTrainer:
    """
    泛化训练器
    
    实现参数化强化学习的完整训练流程，支持：
    - 多约束泛化训练
    - 课程学习
    - 性能监控和早停
    - 模型保存和恢复
    - 详细的训练日志
    """
    
    def __init__(
        self,
        env: ParameterizedPixelAntennaEnv,
        agent_config: AgentConfig,
        training_config: TrainingConfig,
        constraint_manager: ConstraintManager,
        experiment_name: Optional[str] = None
    ):
        """
        初始化训练器
        
        Args:
            env: 参数化环境
            agent_config: 智能体配置
            training_config: 训练配置
            constraint_manager: 约束管理器
            experiment_name: 实验名称
        """
        self.env = env
        self.agent_config = agent_config
        self.training_config = training_config
        self.constraint_manager = constraint_manager
        
        # 实验配置
        self.experiment_name = experiment_name or training_config.experiment_name
        self.experiment_dir = Path(training_config.checkpoint_dir) / self.experiment_name
        self.log_dir = Path(training_config.log_dir) / self.experiment_name
        
        # 创建目录
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.agent = GeneralizedPPOAgent(agent_config)
        self.constraint_sampler = ConstraintSampler(
            constraint_manager=constraint_manager,
            config=training_config.sampling_config
        )
        
        # 课程学习
        if training_config.curriculum_learning:
            self.curriculum_scheduler = CurriculumScheduler(
                stages=training_config.curriculum_stages,
                total_episodes=training_config.total_episodes,
                performance_window=training_config.performance_window,
                min_stage_episodes=training_config.stage_episodes,
                difficulty_progression=training_config.difficulty_progression
            )
        else:
            self.curriculum_scheduler = None
        
        # 训练状态
        self.current_episode = 0
        self.current_constraint = None
        self.constraint_change_counter = 0
        
        # 性能监控
        self.episode_rewards = deque(maxlen=training_config.performance_window)
        self.episode_lengths = deque(maxlen=training_config.performance_window)
        self.success_rates = deque(maxlen=training_config.performance_window)
        self.constraint_performance = defaultdict(list)
        
        # 早停
        self.best_performance = -float('inf')
        self.patience_counter = 0
        self.early_stopped = False
        
        # 日志和监控
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'constraint_changes': [],
            'model_updates': [],
            'evaluation_results': []
        }
        
        # 保存配置
        self._save_configs()
        
        logger.info(f"初始化泛化训练器: {self.experiment_name}")
        logger.info(f"实验目录: {self.experiment_dir}")
        logger.info(f"日志目录: {self.log_dir}")
    
    def _save_configs(self):
        """保存训练配置"""
        config_dir = self.experiment_dir / "configs"
        config_dir.mkdir(exist_ok=True)
        
        # 保存训练配置
        self.training_config.save(str(config_dir / "training_config.json"))
        
        # 保存智能体配置
        with open(config_dir / "agent_config.json", 'w') as f:
            json.dump(self.agent_config.to_dict(), f, indent=2)
        
        # 保存约束信息
        constraint_info = {
            'total_constraints': len(self.constraint_manager.constraints),
            'constraint_groups': len(self.constraint_manager.constraint_groups),
            'sampling_config': self.training_config.sampling_config.to_dict()
        }
        with open(config_dir / "constraint_info.json", 'w') as f:
            json.dump(constraint_info, f, indent=2)
    
    def train(self) -> Dict[str, Any]:
        """
        开始训练
        
        Returns:
            训练结果统计
        """
        logger.info("开始泛化训练...")
        start_time = time.time()
        
        try:
            # 训练循环
            while (self.current_episode < self.training_config.total_episodes and 
                   not self.early_stopped):
                
                # 更新约束
                if self._should_change_constraint():
                    self._update_constraint()
                
                # 训练一个回合
                episode_result = self._train_episode()
                
                # 更新统计
                self._update_statistics(episode_result)
                
                # 课程学习更新
                if self.curriculum_scheduler:
                    stage_changed = self.curriculum_scheduler.update(
                        self.current_episode,
                        episode_result['reward'],
                        episode_result['success']
                    )
                    if stage_changed:
                        self._log_stage_change()
                
                # 模型更新
                if self.current_episode % self.training_config.update_frequency == 0:
                    self._update_agent()
                
                # 评估
                if self.current_episode % self.training_config.eval_frequency == 0:
                    self._evaluate()
                
                # 保存模型
                if self.current_episode % self.training_config.save_frequency == 0:
                    self._save_checkpoint()
                
                # 日志记录
                if self.current_episode % self.training_config.log_frequency == 0:
                    self._log_progress()
                
                # 早停检查
                if self.training_config.early_stopping:
                    self._check_early_stopping()
                
                self.current_episode += 1
            
            # 训练完成
            training_time = time.time() - start_time
            final_results = self._finalize_training(training_time)
            
            logger.info(f"训练完成! 总时间: {training_time:.2f}秒")
            return final_results
            
        except KeyboardInterrupt:
            logger.info("训练被用户中断")
            return self._finalize_training(time.time() - start_time)
        except Exception as e:
            logger.error(f"训练过程中出现错误: {e}")
            raise
        finally:
            self.writer.close()
    
    def _should_change_constraint(self) -> bool:
        """判断是否需要更换约束"""
        return (
            self.current_constraint is None or
            self.constraint_change_counter >= self.training_config.constraint_change_frequency
        )
    
    def _update_constraint(self):
        """更新当前约束"""
        if self.curriculum_scheduler:
            # 使用课程学习采样
            constraints = self.curriculum_scheduler.sample_constraints(
                self.constraint_sampler, num_constraints=1
            )
        else:
            # 使用标准采样
            constraints = self.constraint_sampler.sample_constraints(1)
        
        self.current_constraint = constraints[0]
        self.constraint_change_counter = 0
        
        # 更新环境约束
        self.env.set_constraint(self.current_constraint)
        
        # 记录约束变更
        self.training_stats['constraint_changes'].append({
            'episode': self.current_episode,
            'constraint_id': id(self.current_constraint),
            'constraint_summary': self._get_constraint_summary(self.current_constraint)
        })
        
        logger.debug(f"更新约束 (回合 {self.current_episode}): {self._get_constraint_summary(self.current_constraint)}")
    
    def _get_constraint_summary(self, constraint: ConstraintConfig) -> Dict[str, Any]:
        """获取约束摘要信息"""
        return {
            'frequency_range': (constraint.freq_low, constraint.freq_high),
            'target_s11': constraint.target_s11,
            'tolerance': constraint.tolerance,
            'weight': constraint.weight,
            'name': constraint.name
        }
    
    def _train_episode(self) -> Dict[str, Any]:
        """训练一个回合"""
        state, info = self.env.reset()
        # 使用环境观测中的约束向量部分，而不是单独的to_vector()
        constraint_vector = state[-self.env.constraint_vector_dim:]
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        success = False
        
        while not done and episode_length < self.training_config.max_steps_per_episode:
            # 智能体选择动作
            action, log_prob, value = self.agent.select_action(state, constraint_vector)
            
            # 环境步进 - 处理新的gymnasium API返回值
            step_result = self.env.step(action)
            if len(step_result) == 5:
                # 新的gymnasium API: (obs, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                # 旧的gym API: (obs, reward, done, info)
                next_state, reward, done, info = step_result
            
            # 存储经验
            self.agent.buffer.store(
                state, action, constraint_vector, reward, 
                value, log_prob, done
            )
            
            # 更新状态
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # 检查成功条件
            if info.get('success', False):
                success = True
        
        # 记录约束性能
        constraint_id = id(self.current_constraint)
        self.constraint_performance[constraint_id].append(episode_reward)
        
        self.constraint_change_counter += 1
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'success': success,
            'constraint_id': constraint_id,
            'info': info
        }
    
    def _update_statistics(self, episode_result: Dict[str, Any]):
        """更新训练统计"""
        self.episode_rewards.append(episode_result['reward'])
        self.episode_lengths.append(episode_result['length'])
        self.success_rates.append(float(episode_result['success']))
        
        # 更新训练统计
        self.training_stats['episode_rewards'].append(episode_result['reward'])
        self.training_stats['episode_lengths'].append(episode_result['length'])
        self.training_stats['success_rates'].append(float(episode_result['success']))
    
    def _update_agent(self):
        """更新智能体"""
        if len(self.agent.buffer.states) > 0:
            update_info = self.agent.update()
            
            # 记录更新信息
            self.training_stats['model_updates'].append({
                'episode': self.current_episode,
                'update_info': update_info
            })
            
            # 记录到TensorBoard
            if update_info:
                for key, value in update_info.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'agent/{key}', value, self.current_episode)
    
    def _evaluate(self):
        """评估智能体性能"""
        logger.info(f"开始评估 (回合 {self.current_episode})...")
        
        # 采样评估约束
        eval_constraints = self.constraint_sampler.sample_constraints(
            self.training_config.eval_constraints_num
        )
        
        eval_results = []
        
        for constraint in eval_constraints:
            constraint_results = []
            
            for _ in range(self.training_config.eval_episodes_per_constraint):
                # 设置环境约束
                self.env.set_constraint(constraint)
                
                # 运行评估回合
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple):
                    # 新的gymnasium API: (obs, info)
                    state, _ = reset_result
                else:
                    # 旧的gym API: obs
                    state = reset_result
                # 使用环境观测中的约束向量部分，而不是单独的to_vector()
                constraint_vector = state[-self.env.constraint_vector_dim:]
                
                episode_reward = 0.0
                episode_length = 0
                done = False
                
                while not done and episode_length < self.training_config.max_steps_per_episode:
                    # 使用确定性策略
                    with torch.no_grad():
                        action = self.agent.select_action(
                            state, constraint_vector, deterministic=True
                        )[0]
                    
                    # 环境步进 - 处理新的gymnasium API返回值
                    step_result = self.env.step(action)
                    if len(step_result) == 5:
                        # 新的gymnasium API: (obs, reward, terminated, truncated, info)
                        next_state, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        # 旧的gym API: (obs, reward, done, info)
                        next_state, reward, done, info = step_result
                    
                    state = next_state
                    episode_reward += reward
                    episode_length += 1
                
                constraint_results.append({
                    'reward': episode_reward,
                    'length': episode_length,
                    'success': info.get('success', False)
                })
            
            # 计算约束平均性能
            avg_reward = np.mean([r['reward'] for r in constraint_results])
            avg_length = np.mean([r['length'] for r in constraint_results])
            success_rate = np.mean([r['success'] for r in constraint_results])
            
            eval_results.append({
                'constraint_summary': self._get_constraint_summary(constraint),
                'avg_reward': avg_reward,
                'avg_length': avg_length,
                'success_rate': success_rate,
                'episodes': constraint_results
            })
        
        # 计算总体评估结果
        overall_reward = np.mean([r['avg_reward'] for r in eval_results])
        overall_success_rate = np.mean([r['success_rate'] for r in eval_results])
        
        # 记录评估结果
        eval_summary = {
            'episode': self.current_episode,
            'overall_reward': overall_reward,
            'overall_success_rate': overall_success_rate,
            'num_constraints': len(eval_constraints),
            'detailed_results': eval_results
        }
        
        self.training_stats['evaluation_results'].append(eval_summary)
        
        # 记录到TensorBoard
        self.writer.add_scalar('eval/overall_reward', overall_reward, self.current_episode)
        self.writer.add_scalar('eval/overall_success_rate', overall_success_rate, self.current_episode)
        
        # 更新最佳性能
        if overall_reward > self.best_performance:
            self.best_performance = overall_reward
            self.patience_counter = 0
            
            # 保存最佳模型
            self._save_best_model()
            logger.info(f"新的最佳性能: {overall_reward:.4f}")
        else:
            self.patience_counter += 1
        
        logger.info(
            f"评估完成 - 平均奖励: {overall_reward:.4f}, "
            f"成功率: {overall_success_rate:.2%}"
        )
    
    def _log_stage_change(self):
        """记录阶段变更"""
        if self.curriculum_scheduler:
            progress = self.curriculum_scheduler.get_progress()
            logger.info(
                f"课程学习阶段变更: {progress['current_stage']}/{progress['total_stages']}, "
                f"难度: {progress['current_difficulty']:.2f}, "
                f"策略: {progress['current_strategy']}"
            )
            
            # 记录到TensorBoard
            self.writer.add_scalar('curriculum/stage', progress['current_stage'], self.current_episode)
            self.writer.add_scalar('curriculum/difficulty', progress['current_difficulty'], self.current_episode)
    
    def _log_progress(self):
        """记录训练进度"""
        if len(self.episode_rewards) > 0:
            avg_reward = np.mean(list(self.episode_rewards))
            avg_length = np.mean(list(self.episode_lengths))
            success_rate = np.mean(list(self.success_rates))
            
            logger.info(
                f"回合 {self.current_episode}: "
                f"平均奖励={avg_reward:.4f}, "
                f"平均长度={avg_length:.1f}, "
                f"成功率={success_rate:.2%}"
            )
            
            # 记录到TensorBoard
            self.writer.add_scalar('train/avg_reward', avg_reward, self.current_episode)
            self.writer.add_scalar('train/avg_length', avg_length, self.current_episode)
            self.writer.add_scalar('train/success_rate', success_rate, self.current_episode)
            
            # 课程学习进度
            if self.curriculum_scheduler:
                progress = self.curriculum_scheduler.get_progress()
                self.writer.add_scalar('curriculum/overall_progress', progress['overall_progress'], self.current_episode)
                self.writer.add_scalar('curriculum/stage_progress', progress['stage_progress'], self.current_episode)
    
    def _check_early_stopping(self):
        """检查早停条件"""
        if self.patience_counter >= self.training_config.patience:
            logger.info(
                f"触发早停: {self.patience_counter} 回合无改进 "
                f"(阈值: {self.training_config.patience})"
            )
            self.early_stopped = True
    
    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint_path = self.experiment_dir / f"checkpoint_episode_{self.current_episode}.pt"
        
        checkpoint = {
            'episode': self.current_episode,
            'agent_state_dict': self.agent.get_state_dict(),
            'training_stats': self.training_stats,
            'best_performance': self.best_performance,
            'patience_counter': self.patience_counter,
            'current_constraint': self.current_constraint,
            'constraint_change_counter': self.constraint_change_counter
        }
        
        if self.curriculum_scheduler:
            checkpoint['curriculum_state'] = self.curriculum_scheduler.get_statistics()
        
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"保存检查点: {checkpoint_path}")
    
    def _save_best_model(self):
        """保存最佳模型"""
        if not hasattr(self, 'best_model_path'):
            self.best_model_path = self.experiment_dir / "best_model.pth"
        
        # 使用agent的save方法保存模型
        self.agent.save(str(self.best_model_path))
        
        # 保存额外的训练信息
        best_info = {
            'episode': self.current_episode,
            'best_reward': self.best_performance,
            'timestamp': time.time()
        }
        
        with open(self.experiment_dir / "best_model_info.json", 'w') as f:
            json.dump(best_info, f, indent=2)
        
        logger.info(f"最佳模型已保存 (奖励: {self.best_performance:.4f})")
    
    def _finalize_training(self, training_time: float) -> Dict[str, Any]:
        """完成训练并返回结果"""
        # 最终评估
        logger.info("进行最终评估...")
        self._evaluate()
        
        # 保存最终模型
        final_model_path = self.experiment_dir / "final_model.pt"
        torch.save({
            'episode': self.current_episode,
            'agent_state_dict': self.agent.get_state_dict(),
            'agent_config': self.agent_config.to_dict(),
            'training_config': self.training_config.to_dict(),
            'training_stats': self.training_stats
        }, final_model_path)
        
        # 保存训练统计
        stats_path = self.experiment_dir / "training_stats.json"
        with open(stats_path, 'w') as f:
            # 转换numpy类型为Python原生类型
            serializable_stats = self._make_serializable(self.training_stats)
            json.dump(serializable_stats, f, indent=2)
        
        # 生成训练报告
        final_results = {
            'experiment_name': self.experiment_name,
            'total_episodes': self.current_episode,
            'training_time': training_time,
            'best_performance': self.best_performance,
            'early_stopped': self.early_stopped,
            'final_avg_reward': np.mean(list(self.episode_rewards)) if self.episode_rewards else 0.0,
            'final_success_rate': np.mean(list(self.success_rates)) if self.success_rates else 0.0,
            'total_constraint_changes': len(self.training_stats['constraint_changes']),
            'total_model_updates': len(self.training_stats['model_updates']),
            'total_evaluations': len(self.training_stats['evaluation_results'])
        }
        
        if self.curriculum_scheduler:
            curriculum_stats = self.curriculum_scheduler.get_statistics()
            final_results['curriculum_completed'] = self.curriculum_scheduler.is_completed()
            final_results['final_stage'] = curriculum_stats['curriculum_progress']['current_stage']
            final_results['stage_transitions'] = len(curriculum_stats['stage_transitions'])
        
        # 保存最终结果
        results_path = self.experiment_dir / "final_results.json"
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"训练结果已保存到: {self.experiment_dir}")
        return final_results
    
    def _make_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.agent_config.device)
        
        self.current_episode = checkpoint['episode']
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.training_stats = checkpoint['training_stats']
        self.best_performance = checkpoint['best_performance']
        self.patience_counter = checkpoint['patience_counter']
        self.current_constraint = checkpoint.get('current_constraint')
        self.constraint_change_counter = checkpoint.get('constraint_change_counter', 0)
        
        if 'curriculum_state' in checkpoint and self.curriculum_scheduler:
            # 恢复课程学习状态需要额外处理
            pass
        
        logger.info(f"从检查点恢复训练: 回合 {self.current_episode}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        summary = {
            'experiment_name': self.experiment_name,
            'current_episode': self.current_episode,
            'total_episodes': self.training_config.total_episodes,
            'progress': self.current_episode / self.training_config.total_episodes,
            'best_performance': self.best_performance,
            'early_stopped': self.early_stopped
        }
        
        if len(self.episode_rewards) > 0:
            summary.update({
                'recent_avg_reward': np.mean(list(self.episode_rewards)),
                'recent_success_rate': np.mean(list(self.success_rates)),
                'recent_avg_length': np.mean(list(self.episode_lengths))
            })
        
        if self.curriculum_scheduler:
            summary['curriculum_progress'] = self.curriculum_scheduler.get_progress()
        
        return summary