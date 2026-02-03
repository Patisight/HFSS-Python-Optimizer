"""
简化的PPO智能体

这是一个简化版本的PPO智能体，去除了复杂的约束处理和泛化功能，
只保留基本的PPO算法实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)

class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 动作标准差（可学习参数）
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        mean = self.network(state)
        std = torch.exp(self.log_std)
        return mean, std

class ValueNetwork(nn.Module):
    """价值网络"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(state)

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done, log_prob, value):
        """添加经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done, log_prob, value)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """采样经验"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones, log_probs, values = zip(*batch)
        
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.BoolTensor(dones),
            torch.FloatTensor(log_probs),
            torch.FloatTensor(values)
        )
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer = []
        self.position = 0

class SimplePPOAgent:
    """
    简化的PPO智能体
    
    实现基本的PPO算法，包括策略网络、价值网络和经验回放。
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        buffer_size: int = 2048
    ):
        """
        初始化PPO智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            lr: 学习率
            gamma: 折扣因子
            eps_clip: PPO裁剪参数
            k_epochs: 更新轮数
            buffer_size: 缓冲区大小
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # 创建网络
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        
        # 创建优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # 创建经验缓冲区
        self.buffer = ReplayBuffer(buffer_size)
        
        logger.info(f"简化PPO智能体初始化完成 - 状态维度: {state_dim}, 动作维度: {action_dim}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        选择动作
        
        Args:
            state: 当前状态
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
            value: 状态价值
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            # 获取动作分布
            mean, std = self.policy_net(state_tensor)
            
            # 获取状态价值
            value = self.value_net(state_tensor)
            
            if deterministic:
                # 确定性策略：使用均值
                action = mean
                log_prob = 0.0
            else:
                # 随机策略：从正态分布采样
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.squeeze(0).numpy(), log_prob.item(), value.squeeze(0).item()
    
    def store_experience(self, state, action, reward, next_state, done, log_prob, value):
        """存储经验"""
        self.buffer.push(state, action, reward, next_state, done, log_prob, value)
    
    def update(self) -> Dict[str, float]:
        """更新网络"""
        if len(self.buffer) < 64:  # 最小批次大小
            return {}
        
        # 计算优势和回报
        states, actions, rewards, next_states, dones, old_log_probs, old_values = self.buffer.sample(len(self.buffer))
        
        # 计算折扣回报
        returns = self._compute_returns(rewards, dones)
        
        # 计算优势
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        policy_losses = []
        value_losses = []
        
        for _ in range(self.k_epochs):
            # 重新计算动作概率和价值
            mean, std = self.policy_net(states)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            new_values = self.value_net(states).squeeze()
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = F.mse_loss(new_values, returns)
            
            # 更新策略网络
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # 更新价值网络
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
        
        # 清空缓冲区
        self.buffer.clear()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses)
        }
    
    def _compute_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """计算折扣回报"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, filepath)
        logger.info(f"模型已保存到: {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        logger.info(f"模型已从 {filepath} 加载")