"""
泛化PPO智能体

实现参数化强化学习智能体，支持条件策略网络π(a|s,c)和泛化能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json
from collections import deque
import random

from .agent_config import AgentConfig
from .policy_networks import ConditionalPolicyNetwork, ValueNetwork
from ..config.constraint_config import ConstraintConfig, ConstraintGroup

class PPOBuffer:
    """PPO经验缓冲区"""
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, constraint_dim: int):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
        
        # 缓冲区数据
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.constraints = np.zeros((buffer_size, constraint_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        
        # GAE计算用
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
    
    def store(self, state: np.ndarray, action: np.ndarray, constraint: np.ndarray,
              reward: float, value: float, log_prob: float, done: bool):
        """存储经验"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.constraints[self.ptr] = constraint
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """计算GAE优势估计"""
        advantages = np.zeros_like(self.rewards)
        last_gae_lam = 0
        
        for step in reversed(range(self.size)):
            if step == self.size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]
            
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            advantages[step] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        self.advantages[:self.size] = advantages[:self.size]
        self.returns[:self.size] = advantages[:self.size] + self.values[:self.size]
    
    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """获取训练批次"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'constraints': torch.FloatTensor(self.constraints[indices]),
            'old_log_probs': torch.FloatTensor(self.log_probs[indices]),
            'advantages': torch.FloatTensor(self.advantages[indices]),
            'returns': torch.FloatTensor(self.returns[indices])
        }
    
    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0

class GeneralizedPPOAgent:
    """
    泛化PPO智能体
    
    实现参数化强化学习，支持条件策略网络π(a|s,c)和多约束泛化
    """
    
    def __init__(self, 
                 config: AgentConfig,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化智能体
        
        Args:
            config: 智能体配置
            device: 计算设备
        """
        self.config = config
        self.device = torch.device(device)
        
        # 验证配置
        config.validate()
        
        # 创建网络
        self.policy_net = ConditionalPolicyNetwork(config).to(self.device)
        self.value_net = ValueNetwork(config).to(self.device)
        
        # 优化器
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.learning_rate,
            weight_decay=config.l2_reg
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(),
            lr=config.value_learning_rate,
            weight_decay=config.l2_reg
        )
        
        # 经验缓冲区
        self.buffer = PPOBuffer(
            config.buffer_size,
            config.state_dim,
            config.action_dim,
            config.constraint_dim
        )
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': deque(maxlen=100),
            'episode_lengths': deque(maxlen=100),
            'policy_losses': deque(maxlen=100),
            'value_losses': deque(maxlen=100),
            'constraint_performance': {}
        }
        
        # 日志配置
        self.logger = logging.getLogger(__name__)
        
        # 当前约束（用于训练）
        self.current_constraint = None
    
    def select_action(self, state: np.ndarray, constraint: np.ndarray, 
                     deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        选择动作
        
        Args:
            state: 环境状态
            constraint: 约束向量
            deterministic: 是否确定性选择
            
        Returns:
            action: 选择的动作
            log_prob: 动作对数概率
            value: 状态价值估计
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            constraint_tensor = torch.FloatTensor(constraint).unsqueeze(0).to(self.device)
            
            if deterministic:
                # 确定性选择（用于评估）
                action_probs = self.policy_net(state_tensor, constraint_tensor)
                # 选择概率最高的动作索引
                action_idx = torch.argmax(action_probs, dim=-1)
                log_prob = 0.0
            else:
                # 随机采样（用于训练）
                action_probs = self.policy_net(state_tensor, constraint_tensor)
                # 将多维概率转换为单一动作选择
                # 使用softmax将概率归一化，然后采样单一动作索引
                action_probs_normalized = F.softmax(action_probs, dim=-1)
                dist = torch.distributions.Categorical(action_probs_normalized)
                action_idx = dist.sample()
                log_prob = dist.log_prob(action_idx).item()
            
            # 价值估计
            value = self.value_net(state_tensor, constraint_tensor).item()
            
            return action_idx.cpu().numpy().squeeze(), log_prob, value
    
    def update(self) -> Dict[str, float]:
        """
        PPO更新
        
        Returns:
            训练统计信息
        """
        if self.buffer.size < self.config.batch_size:
            return {}
        
        # 计算GAE
        with torch.no_grad():
            last_state = torch.FloatTensor(self.buffer.states[self.buffer.size-1]).unsqueeze(0).to(self.device)
            last_constraint = torch.FloatTensor(self.buffer.constraints[self.buffer.size-1]).unsqueeze(0).to(self.device)
            last_value = self.value_net(last_state, last_constraint).item()
        
        self.buffer.compute_gae(last_value, self.config.gamma, self.config.gae_lambda)
        
        # 训练统计
        policy_losses = []
        value_losses = []
        
        # PPO更新轮次
        for epoch in range(self.config.ppo_epochs):
            # 获取训练批次
            batch = self.buffer.get_batch(self.config.batch_size)
            
            # 移动到设备
            for key in batch:
                batch[key] = batch[key].to(self.device)
            
            # 计算当前策略的对数概率和熵
            current_log_probs, entropy = self.policy_net.evaluate_actions(
                batch['states'], batch['constraints'], batch['actions']
            )
            
            # 计算比率
            ratio = torch.exp(current_log_probs - batch['old_log_probs'])
            
            # 标准化优势
            advantages = batch['advantages']
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO裁剪损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 熵损失
            entropy_loss = -self.config.entropy_coef * entropy.mean()
            
            # 总策略损失
            total_policy_loss = policy_loss + entropy_loss
            
            # 约束一致性损失（可选）
            if self.config.constraint_consistency > 0:
                consistency_loss = self._compute_constraint_consistency_loss(batch)
                total_policy_loss += self.config.constraint_consistency * consistency_loss
            
            # 价值损失
            current_values = self.value_net(batch['states'], batch['constraints']).squeeze()
            value_loss = nn.MSELoss()(current_values, batch['returns'])
            
            # 更新策略网络
            self.policy_optimizer.zero_grad()
            total_policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.max_grad_norm)
            self.policy_optimizer.step()
            
            # 更新价值网络
            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.config.max_grad_norm)
            self.value_optimizer.step()
            
            # 记录损失
            policy_losses.append(total_policy_loss.item())
            value_losses.append(value_loss.item())
        
        # 清空缓冲区
        self.buffer.clear()
        
        # 更新统计
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        
        self.training_stats['policy_losses'].append(avg_policy_loss)
        self.training_stats['value_losses'].append(avg_value_loss)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'buffer_size': self.buffer.size
        }
    
    def _compute_constraint_consistency_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算约束一致性损失"""
        # 简单实现：相似约束应该产生相似的策略
        constraints = batch['constraints']
        states = batch['states']
        
        # 计算约束相似性
        constraint_sim = torch.mm(constraints, constraints.t())
        constraint_sim = torch.softmax(constraint_sim / 0.1, dim=-1)
        
        # 计算策略相似性
        action_probs = self.policy_net(states, constraints)
        policy_sim = torch.mm(action_probs, action_probs.t())
        policy_sim = torch.softmax(policy_sim / 0.1, dim=-1)
        
        # KL散度损失
        consistency_loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log(policy_sim + 1e-8),
            constraint_sim
        )
        
        return consistency_loss
    
    def train_episode(self, env, constraint: ConstraintConfig, max_steps: int = 1000) -> Dict[str, Any]:
        """
        训练一个回合
        
        Args:
            env: 环境实例
            constraint: 约束配置
            max_steps: 最大步数
            
        Returns:
            回合统计信息
        """
        self.current_constraint = constraint
        constraint_vector = constraint.to_vector()
        
        # 重置环境
        state, _ = env.reset()
        env.set_constraint(constraint)
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # 选择动作
            action, log_prob, value = self.select_action(state, constraint_vector)
            
            # 执行动作
            next_state, reward, done, truncated, info = env.step(action)
            
            # 存储经验
            self.buffer.store(
                state, action, constraint_vector,
                reward, value, log_prob, done or truncated
            )
            
            # 更新统计
            episode_reward += reward
            episode_length += 1
            
            # 更新状态
            state = next_state
            
            # 检查终止条件
            if done or truncated:
                break
        
        # 记录回合统计
        self.training_stats['episode_rewards'].append(episode_reward)
        self.training_stats['episode_lengths'].append(episode_length)
        
        # 更新约束性能统计
        constraint_key = f"{constraint.name}"
        if constraint_key not in self.training_stats['constraint_performance']:
            self.training_stats['constraint_performance'][constraint_key] = deque(maxlen=50)
        self.training_stats['constraint_performance'][constraint_key].append(episode_reward)
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'constraint': constraint.name,
            'info': info
        }
    
    def evaluate(self, env, constraints: List[ConstraintConfig], 
                n_episodes_per_constraint: int = 10) -> Dict[str, Any]:
        """
        评估智能体性能
        
        Args:
            env: 环境实例
            constraints: 约束列表
            n_episodes_per_constraint: 每个约束的评估回合数
            
        Returns:
            评估结果
        """
        results = {}
        
        for constraint in constraints:
            constraint_results = []
            constraint_vector = constraint.to_vector()
            
            for episode in range(n_episodes_per_constraint):
                # 重置环境
                state, _ = env.reset()
                env.set_constraint(constraint)
                
                episode_reward = 0
                episode_length = 0
                
                done = False
                while not done and episode_length < 1000:
                    # 确定性选择动作
                    action, _, _ = self.select_action(state, constraint_vector, deterministic=True)
                    
                    # 执行动作
                    state, reward, done, truncated, info = env.step(action)
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    if truncated:
                        break
                
                constraint_results.append({
                    'reward': episode_reward,
                    'length': episode_length,
                    'info': info
                })
            
            # 计算统计信息
            rewards = [r['reward'] for r in constraint_results]
            results[constraint.name] = {
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'min_reward': np.min(rewards),
                'max_reward': np.max(rewards),
                'episodes': constraint_results
            }
        
        return results
    
    def save(self, filepath: str):
        """保存模型"""
        save_dict = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'config': self.config.to_dict(),
            'training_stats': dict(self.training_stats)
        }
        
        torch.save(save_dict, filepath)
        self.logger.info(f"模型已保存到: {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        # 恢复训练统计
        if 'training_stats' in checkpoint:
            for key, value in checkpoint['training_stats'].items():
                if isinstance(value, list):
                    self.training_stats[key] = deque(value, maxlen=100)
                else:
                    self.training_stats[key] = value
        
        self.logger.info(f"模型已从 {filepath} 加载")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        stats = {}
        
        if self.training_stats['episode_rewards']:
            stats['mean_episode_reward'] = np.mean(self.training_stats['episode_rewards'])
            stats['std_episode_reward'] = np.std(self.training_stats['episode_rewards'])
            stats['mean_episode_length'] = np.mean(self.training_stats['episode_lengths'])
        
        if self.training_stats['policy_losses']:
            stats['mean_policy_loss'] = np.mean(self.training_stats['policy_losses'])
            stats['mean_value_loss'] = np.mean(self.training_stats['value_losses'])
        
        # 约束性能统计
        constraint_stats = {}
        for constraint_name, rewards in self.training_stats['constraint_performance'].items():
            if rewards:
                constraint_stats[constraint_name] = {
                    'mean_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'episodes': len(rewards)
                }
        stats['constraint_performance'] = constraint_stats
        
        return stats