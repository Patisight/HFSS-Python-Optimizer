"""
泛化PPO智能体 - 支持参数化策略学习和约束适应
基于Stable Baselines3实现，支持动态约束注入和元学习

核心特性:
1. 参数化策略: 将约束向量作为策略输入，实现条件策略学习
2. 自适应网络: 动态调整网络结构以适应不同约束复杂度
3. 经验重放: 支持多约束经验的有效利用
4. 元学习支持: 为Meta-RL提供基础架构
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
import logging
from dataclasses import dataclass
import json
from pathlib import Path

# 导入环境和约束模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.parameterized_pixel_env import ParameterizedPixelAntennaEnv
from constraint.types import ConstraintConfig
from constraint.constraint_manager import ConstraintGroup

@dataclass
class AgentConfig:
    """智能体配置"""
    # 网络结构
    policy_layers: List[int] = None  # [256, 256, 128]
    value_layers: List[int] = None   # [256, 256, 128]
    constraint_embedding_dim: int = 64
    
    # PPO参数
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    
    # 泛化相关
    constraint_dropout: float = 0.1  # 约束向量dropout
    domain_adaptation: bool = True   # 启用域适应
    meta_learning: bool = False      # 启用元学习
    
    def __post_init__(self):
        if self.policy_layers is None:
            self.policy_layers = [256, 256, 128]
        if self.value_layers is None:
            self.value_layers = [256, 256, 128]

class ConstraintAwareFeatureExtractor(BaseFeaturesExtractor):
    """
    约束感知特征提取器
    
    将观测空间分解为：像素配置 + S11数据 + 物理特征 + 约束向量
    并进行专门的特征提取和融合
    """
    
    def __init__(self, 
                 observation_space: gym.Space,
                 pixel_dim: int = 99,  # 修改为99个像素
                 s11_dim: int = 20,
                 physics_dim: int = 4,
                 constraint_dim: int = 3,
                 features_dim: int = 256):
        """
        初始化特征提取器
        
        Args:
            observation_space: 观测空间
            pixel_dim: 像素配置维度
            s11_dim: S11数据维度
            physics_dim: 物理特征维度
            constraint_dim: 约束向量维度
            features_dim: 输出特征维度
        """
        super().__init__(observation_space, features_dim)
        
        self.pixel_dim = pixel_dim
        self.s11_dim = s11_dim
        self.physics_dim = physics_dim
        self.constraint_dim = constraint_dim
        
        # 像素配置编码器
        self.pixel_encoder = nn.Sequential(
            nn.Linear(pixel_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # S11数据编码器（1D CNN）
        self.s11_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(32 * 8, 64),
            nn.ReLU()
        )
        
        # 物理特征编码器
        self.physics_encoder = nn.Sequential(
            nn.Linear(physics_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # 约束向量编码器
        self.constraint_encoder = nn.Sequential(
            nn.Linear(constraint_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 特征融合网络
        total_encoded_dim = 64 + 64 + 32 + 64  # pixel + s11 + physics + constraint
        self.fusion_network = nn.Sequential(
            nn.Linear(total_encoded_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size = observations.shape[0]
        
        # 分解观测
        pixel_obs = observations[:, :self.pixel_dim]
        s11_obs = observations[:, self.pixel_dim:self.pixel_dim + self.s11_dim]
        physics_obs = observations[:, self.pixel_dim + self.s11_dim:self.pixel_dim + self.s11_dim + self.physics_dim]
        constraint_obs = observations[:, -self.constraint_dim:]
        
        # 编码各部分
        pixel_features = self.pixel_encoder(pixel_obs)
        
        # S11数据需要reshape为(batch, channel, length)
        s11_features = self.s11_encoder(s11_obs.unsqueeze(1))
        
        physics_features = self.physics_encoder(physics_obs)
        constraint_features = self.constraint_encoder(constraint_obs)
        
        # 特征融合
        combined_features = torch.cat([
            pixel_features, s11_features, physics_features, constraint_features
        ], dim=1)
        
        # 应用dropout和融合网络
        combined_features = self.dropout(combined_features)
        output_features = self.fusion_network(combined_features)
        
        return output_features

class ParameterizedActorCriticPolicy(ActorCriticPolicy):
    """
    参数化Actor-Critic策略
    
    支持约束条件的策略学习
    """
    
    def __init__(self, *args, **kwargs):
        # 提取自定义参数
        self.constraint_embedding_dim = kwargs.pop('constraint_embedding_dim', 64)
        self.constraint_dropout = kwargs.pop('constraint_dropout', 0.1)
        
        super().__init__(*args, **kwargs)
        
    def _build_mlp_extractor(self) -> None:
        """构建MLP特征提取器（使用标准MlpExtractor）"""
        super()._build_mlp_extractor()

class ConstraintAdaptationCallback(BaseCallback):
    """
    约束适应回调
    
    监控不同约束下的性能，并调整训练策略
    """
    
    def __init__(self, 
                 log_interval: int = 100,
                 adaptation_interval: int = 1000,
                 verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.adaptation_interval = adaptation_interval
        
        # 性能统计
        self.constraint_performance: Dict[str, List[float]] = {}
        self.current_constraint = None
        
    def _on_step(self) -> bool:
        """每步回调"""
        # 获取当前约束信息
        if hasattr(self.training_env, 'get_attr'):
            try:
                env_info_fn = self.training_env.get_attr('_get_info')[0]
                env_info = env_info_fn() if callable(env_info_fn) else env_info_fn
                if env_info:
                    if 'constraint_group' in env_info:
                        cg = env_info['constraint_group']
                        self.current_constraint = f"GROUP:{cg['name']}:{cg['n_bands']}"
                    elif 'constraint' in env_info:
                        c = env_info['constraint']
                        self.current_constraint = f"{c['freq_low']:.0f}-{c['freq_high']:.0f}-{c['target_s11']:.1f}"
            except Exception:
                pass
        
        # 记录性能
        if self.num_timesteps % self.log_interval == 0:
            self._log_performance()
        
        # 适应性调整
        if self.num_timesteps % self.adaptation_interval == 0:
            self._adapt_training()
        
        return True
        
    def _log_performance(self):
        """记录性能统计"""
        if self.current_constraint is None:
            return
            
        # 获取最近的奖励
        if len(self.model.ep_info_buffer) > 0:
            recent = list(self.model.ep_info_buffer)
            recent_rewards = [ep_info['r'] for ep_info in recent[-10:]]
            avg_reward = np.mean(recent_rewards)
            
            if self.current_constraint not in self.constraint_performance:
                self.constraint_performance[self.current_constraint] = []
            self.constraint_performance[self.current_constraint].append(avg_reward)
            
            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: Constraint {self.current_constraint}, Avg Reward: {avg_reward:.2f}")
                
    def _adapt_training(self):
        """自适应训练调整"""
        if len(self.constraint_performance) < 2:
            return
            
        # 分析不同约束的性能差异
        performance_stats = {}
        for constraint, rewards in self.constraint_performance.items():
            if len(rewards) >= 5:  # 至少5个数据点
                performance_stats[constraint] = {
                    'mean': np.mean(rewards[-10:]),  # 最近10个
                    'std': np.std(rewards[-10:]),
                    'trend': np.mean(rewards[-5:]) - np.mean(rewards[-10:-5]) if len(rewards) >= 10 else 0
                }
                
        if self.verbose > 0:
            print(f"Performance adaptation at step {self.num_timesteps}:")
            for constraint, stats in performance_stats.items():
                print(f"  {constraint}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, trend={stats['trend']:.2f}")

class GeneralizedPPOAgent:
    """
    泛化PPO智能体
    
    支持多约束学习和快速适应
    """
    
    def __init__(self, 
                 env: ParameterizedPixelAntennaEnv,
                 config: Optional[AgentConfig] = None,
                 model_save_path: Optional[str] = None):
        """
        初始化泛化PPO智能体
        
        Args:
            env: 参数化环境
            config: 智能体配置
            model_save_path: 模型保存路径
        """
        self.env = env
        self.config = config or AgentConfig()
        self.model_save_path = model_save_path
        
        # 创建PPO模型
        self.model = self._create_model()
        
        # 训练统计
        self.training_stats = {
            'total_timesteps': 0,
            'episodes': 0,
            'constraint_history': [],
            'performance_history': []
        }
        
        # 日志配置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _create_model(self) -> PPO:
        """创建PPO模型"""
        # 策略参数
        policy_kwargs = {
            'features_extractor_class': ConstraintAwareFeatureExtractor,
            'features_extractor_kwargs': {
                'pixel_dim': self.env.pixel_count,
                's11_dim': self.env.freq_samples,
                'physics_dim': 4,
                'constraint_dim': getattr(self.env, 'constraint_vector_dim', 3),
                'features_dim': 256
            },
            'net_arch': {
                'pi': self.config.policy_layers,
                'vf': self.config.value_layers
            },
            'constraint_embedding_dim': self.config.constraint_embedding_dim,
            'constraint_dropout': self.config.constraint_dropout
        }
        
        # 创建PPO模型
        model = PPO(
            policy=ParameterizedActorCriticPolicy,
            env=self.env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device='auto'
        )
        
        return model
        
    def train(self, 
              total_timesteps: int,
              constraints: List,  # 支持 ConstraintConfig 或 ConstraintGroup
              constraint_schedule: str = 'random',
              save_interval: int = 10000,
              callback_kwargs: Optional[Dict] = None) -> Dict[str, Any]:
        """
        训练智能体
        
        Args:
            total_timesteps: 总训练步数
            constraints: 约束列表
            constraint_schedule: 约束调度策略 ('random', 'sequential', 'curriculum')
            save_interval: 保存间隔
            callback_kwargs: 回调参数
            
        Returns:
            训练统计信息
        """
        self.logger.info(f"开始训练，总步数: {total_timesteps}, 约束数量: {len(constraints)}")
        
        # 创建回调
        callback_kwargs = callback_kwargs or {}
        callback = ConstraintAdaptationCallback(**callback_kwargs)
        
        # 约束调度器
        constraint_scheduler = self._create_constraint_scheduler(constraints, constraint_schedule)
        
        # 训练循环
        steps_per_constraint = total_timesteps // len(constraints)
        
        for i, constraint in enumerate(constraint_scheduler):
            name = getattr(constraint, 'name', f"constraint_{i}")
            self.logger.info(f"训练约束 {i+1}/{len(constraints)}: {name}")
            
            # 设置环境约束（兼容组）
            self.env.set_constraint(constraint)
            
            # 训练当前约束
            self.model.learn(
                total_timesteps=steps_per_constraint,
                callback=callback,
                reset_num_timesteps=False
            )
            
            # 更新统计
            self.training_stats['constraint_history'].append(name)
            self.training_stats['total_timesteps'] += steps_per_constraint
            
            # 定期保存
            if self.model_save_path:
                save_every = max(1, len(constraints) // 5)
                if (i + 1) % save_every == 0:
                    save_path = f"{self.model_save_path}_checkpoint_{i+1}"
                    self.save_model(save_path)
                
        # 最终保存
        if self.model_save_path:
            self.save_model(self.model_save_path)
            
        self.logger.info("训练完成")
        return self.training_stats
        
    def _create_constraint_scheduler(self, 
                                   constraints: List, 
                                   schedule: str) -> List:
        """创建约束调度器，支持约束组"""
        if schedule == 'random':
            scheduled = constraints.copy()
            np.random.shuffle(scheduled)
            return scheduled
        elif schedule == 'sequential':
            return constraints
        elif schedule == 'curriculum':
            def difficulty_score(c) -> float:
                # 约束组：按内部平均带宽与严格度评分
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
        else:
            raise ValueError(f"未知的约束调度策略: {schedule}")
            
    def evaluate(self, 
                 constraints: List,  # 支持 ConstraintConfig 或 ConstraintGroup
                 n_episodes_per_constraint: int = 10) -> Dict[str, Any]:
        """
        评估智能体性能
        
        Args:
            constraints: 测试约束列表
            n_episodes_per_constraint: 每个约束的测试回合数
            
        Returns:
            评估结果
        """
        self.logger.info(f"开始评估，约束数量: {len(constraints)}")
        
        results = {}
        
        for constraint in constraints:
            name = getattr(constraint, 'name', 'constraint')
            self.logger.info(f"评估约束: {name}")
            # 设置约束
            self.env.set_constraint(constraint)
            
            # 运行测试回合
            episode_rewards = []
            episode_lengths = []
            success_count = 0
            
            for episode in range(n_episodes_per_constraint):
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
                done = False
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    done = terminated or truncated
                    
                    # 检查是否成功
                    if terminated and reward > 0:
                        success_count += 1
                        
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
            # 统计结果
            results[name] = {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'mean_length': np.mean(episode_lengths),
                'success_rate': success_count / n_episodes_per_constraint,
                'constraint_info': (
                    {
                        'bands': [[sc.freq_low/1e9, sc.freq_high/1e9, sc.target_s11] for sc in constraint.constraints]
                    } if isinstance(constraint, ConstraintGroup) else {
                        'freq_range': [constraint.freq_low/1e9, constraint.freq_high/1e9],
                        'target_s11': constraint.target_s11,
                        'tolerance': constraint.tolerance
                    }
                )
            }
            
        # 计算总体统计
        all_rewards = [r['mean_reward'] for r in results.values()]
        all_success_rates = [r['success_rate'] for r in results.values()]
        
        overall_stats = {
            'overall_mean_reward': np.mean(all_rewards),
            'overall_std_reward': np.std(all_rewards),
            'overall_success_rate': np.mean(all_success_rates),
            'constraint_results': results
        }
        
        self.logger.info(f"评估完成，总体成功率: {overall_stats['overall_success_rate']:.2%}")
        return overall_stats
        
    def fine_tune(self, 
                  new_constraints: List,  # 支持 ConstraintConfig 或 ConstraintGroup
                  fine_tune_steps: int = 5000,
                  learning_rate_factor: float = 0.1) -> Dict[str, Any]:
        """
        针对新约束进行微调
        
        Args:
            new_constraints: 新约束列表
            fine_tune_steps: 微调步数
            learning_rate_factor: 学习率缩放因子
            
        Returns:
            微调统计信息
        """
        self.logger.info(f"开始微调，新约束数量: {len(new_constraints)}")
        
        # 降低学习率
        original_lr = self.model.learning_rate
        self.model.learning_rate = original_lr * learning_rate_factor
        
        # 微调每个新约束
        fine_tune_stats = {}
        
        for constraint in new_constraints:
            name = getattr(constraint, 'name', 'constraint')
            self.logger.info(f"微调约束: {name}")
            
            # 设置约束
            self.env.set_constraint(constraint)
            
            # 记录微调前性能
            pre_performance = self._quick_evaluate(constraint, n_episodes=5)
            
            # 微调训练
            self.model.learn(
                total_timesteps=fine_tune_steps,
                reset_num_timesteps=False
            )
            
            # 记录微调后性能
            post_performance = self._quick_evaluate(constraint, n_episodes=5)
            
            fine_tune_stats[name] = {
                'pre_reward': pre_performance['mean_reward'],
                'post_reward': post_performance['mean_reward'],
                'improvement': post_performance['mean_reward'] - pre_performance['mean_reward'],
                'pre_success_rate': pre_performance['success_rate'],
                'post_success_rate': post_performance['success_rate']
            }
            
        # 恢复学习率
        self.model.learning_rate = original_lr
        
        self.logger.info("微调完成")
        return fine_tune_stats
        
    def _quick_evaluate(self, constraint: ConstraintConfig, n_episodes: int = 5) -> Dict[str, float]:
        """快速评估单个约束"""
        self.env.set_constraint(constraint)
        
        rewards = []
        success_count = 0
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
                if terminated and reward > 0:
                    success_count += 1
                    
            rewards.append(episode_reward)
            
        return {
            'mean_reward': np.mean(rewards),
            'success_rate': success_count / n_episodes
        }
        
    def save_model(self, path: str):
        """保存模型"""
        self.model.save(path)
        
        # 保存配置和统计信息
        config_path = f"{path}_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'config': self.config.__dict__,
                'training_stats': self.training_stats
            }, f, indent=2)
            
        self.logger.info(f"模型已保存到: {path}")
        
    def load_model(self, path: str):
        """加载模型"""
        self.model = PPO.load(path, env=self.env)
        
        # 加载配置和统计信息
        config_path = f"{path}_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                data = json.load(f)
                self.training_stats = data.get('training_stats', {})
                
        self.logger.info(f"模型已从文件加载: {path}")
        
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_class': self.model.__class__.__name__,
            'policy_class': self.model.policy.__class__.__name__,
            'total_parameters': sum(p.numel() for p in self.model.policy.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.policy.parameters() if p.requires_grad),
            'config': self.config.__dict__,
            'training_stats': self.training_stats
        }