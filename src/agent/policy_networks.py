"""
策略网络

实现条件策略网络π(a|s,c)和价值网络V(s,c)，支持参数化强化学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np

from .agent_config import AgentConfig, NetworkConfig

class ConstraintEmbedding(nn.Module):
    """约束嵌入网络"""
    
    def __init__(self, constraint_dim: int, embedding_dim: int):
        super().__init__()
        self.constraint_dim = constraint_dim
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Sequential(
            nn.Linear(constraint_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, constraints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            constraints: [batch_size, constraint_dim]
        Returns:
            embedded_constraints: [batch_size, embedding_dim]
        """
        return self.embedding(constraints)

class StateEmbedding(nn.Module):
    """状态嵌入网络"""
    
    def __init__(self, state_dim: int, embedding_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        
        # 检查状态维度是否为完美平方数
        grid_size = int(np.sqrt(state_dim))
        self.is_square = (grid_size * grid_size == state_dim)
        
        if self.is_square:
            # 使用CNN提取空间特征
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 减半
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 再减半
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(4)  # 4x4
            )
            
            # 全连接层
            self.fc = nn.Sequential(
                nn.Linear(256 * 4 * 4, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.ReLU()
            )
        else:
            # 对于非平方维度，使用纯全连接网络
            self.fc = nn.Sequential(
                nn.Linear(state_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.ReLU()
            )
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: [batch_size, state_dim] 或 [batch_size, 1, H, W]
        Returns:
            embedded_states: [batch_size, embedding_dim]
        """
        batch_size = states.shape[0]
        
        if self.is_square and len(states.shape) == 2:
            # 重塑为图像格式
            grid_size = int(np.sqrt(self.state_dim))
            states = states.view(batch_size, 1, grid_size, grid_size)
            
            # CNN特征提取
            features = self.cnn(states)
            features = features.view(batch_size, -1)
            
            # 全连接嵌入
            embedded = self.fc(features)
        else:
            # 对于非平方维度或已经是图像格式的输入，使用全连接网络
            if len(states.shape) > 2:
                states = states.view(batch_size, -1)
            embedded = self.fc(states)
            
        return embedded

class AttentionFusion(nn.Module):
    """注意力融合模块"""
    
    def __init__(self, state_dim: int, constraint_dim: int, hidden_dim: int):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.constraint_proj = nn.Linear(constraint_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, state_emb: torch.Tensor, constraint_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_emb: [batch_size, state_embedding_dim]
            constraint_emb: [batch_size, constraint_embedding_dim]
        Returns:
            fused: [batch_size, hidden_dim]
        """
        # 投影到相同维度
        state_proj = self.state_proj(state_emb).unsqueeze(1)  # [B, 1, H]
        constraint_proj = self.constraint_proj(constraint_emb).unsqueeze(1)  # [B, 1, H]
        
        # 注意力融合
        fused, _ = self.attention(state_proj, constraint_proj, constraint_proj)
        fused = self.norm(fused.squeeze(1))  # [B, H]
        
        return fused

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) 层"""
    
    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(condition_dim, feature_dim)
        self.beta_proj = nn.Linear(condition_dim, feature_dim)
        
    def forward(self, features: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, feature_dim]
            conditions: [batch_size, condition_dim]
        Returns:
            modulated_features: [batch_size, feature_dim]
        """
        gamma = self.gamma_proj(conditions)
        beta = self.beta_proj(conditions)
        return gamma * features + beta

class ConditionalPolicyNetwork(nn.Module):
    """
    条件策略网络 π(a|s,c)
    
    实现参数化强化学习的核心网络，根据状态s和约束c输出动作分布
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.state_embedding = StateEmbedding(
            config.state_dim, 
            config.state_embedding_dim
        )
        self.constraint_embedding = ConstraintEmbedding(
            config.constraint_dim,
            config.constraint_embedding_dim
        )
        
        # 融合层
        if config.fusion_method == "concat":
            fusion_dim = config.state_embedding_dim + config.constraint_embedding_dim
            self.fusion = nn.Identity()
        elif config.fusion_method == "attention":
            fusion_dim = config.state_embedding_dim
            self.fusion = AttentionFusion(
                config.state_embedding_dim,
                config.constraint_embedding_dim,
                fusion_dim
            )
        elif config.fusion_method == "film":
            fusion_dim = config.state_embedding_dim
            self.fusion = FiLMLayer(
                config.state_embedding_dim,
                config.constraint_embedding_dim
            )
        else:
            raise ValueError(f"不支持的融合方法: {config.fusion_method}")
        
        # 策略网络
        layers = []
        input_dim = fusion_dim
        
        for hidden_dim in config.policy_config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if config.policy_config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif config.policy_config.layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(config.get_activation_fn(config.policy_config.activation))
            
            if config.policy_config.dropout_rate > 0:
                layers.append(nn.Dropout(config.policy_config.dropout_rate))
            
            input_dim = hidden_dim
        
        # 输出层 - 每个像素的开/关概率
        layers.append(nn.Linear(input_dim, config.action_dim))
        layers.append(nn.Sigmoid())  # 输出概率
        
        self.policy_head = nn.Sequential(*layers)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化网络权重"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, states: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            states: [batch_size, state_dim]
            constraints: [batch_size, constraint_dim]
            
        Returns:
            action_probs: [batch_size, action_dim] 每个像素的激活概率
        """
        # 嵌入
        state_emb = self.state_embedding(states)
        constraint_emb = self.constraint_embedding(constraints)
        
        # 融合
        if self.config.fusion_method == "concat":
            fused = torch.cat([state_emb, constraint_emb], dim=-1)
        elif self.config.fusion_method == "attention":
            fused = self.fusion(state_emb, constraint_emb)
        elif self.config.fusion_method == "film":
            fused = self.fusion(state_emb, constraint_emb)
        
        # 策略输出
        action_probs = self.policy_head(fused)
        return action_probs
    
    def sample_action(self, states: torch.Tensor, constraints: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作
        
        Args:
            states: [batch_size, state_dim]
            constraints: [batch_size, constraint_dim]
            
        Returns:
            actions: [batch_size, action_dim] 二值动作
            log_probs: [batch_size] 动作的对数概率
        """
        action_probs = self.forward(states, constraints)
        
        # 伯努利采样
        dist = torch.distributions.Bernoulli(action_probs)
        actions = dist.sample()
        
        # 计算对数概率
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        return actions, log_probs
    
    def evaluate_actions(self, states: torch.Tensor, constraints: torch.Tensor, 
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估动作
        
        Args:
            states: [batch_size, state_dim]
            constraints: [batch_size, constraint_dim]
            actions: [batch_size, action_dim]
            
        Returns:
            log_probs: [batch_size] 动作的对数概率
            entropy: [batch_size] 策略熵
        """
        action_probs = self.forward(states, constraints)
        
        # 计算对数概率和熵
        dist = torch.distributions.Bernoulli(action_probs)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, entropy

class ValueNetwork(nn.Module):
    """
    价值网络 V(s,c)
    
    估计在给定状态s和约束c下的状态价值
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        
        # 嵌入层（与策略网络共享）
        self.state_embedding = StateEmbedding(
            config.state_dim,
            config.state_embedding_dim
        )
        self.constraint_embedding = ConstraintEmbedding(
            config.constraint_dim,
            config.constraint_embedding_dim
        )
        
        # 融合层
        if config.fusion_method == "concat":
            fusion_dim = config.state_embedding_dim + config.constraint_embedding_dim
            self.fusion = nn.Identity()
        elif config.fusion_method == "attention":
            fusion_dim = config.state_embedding_dim
            self.fusion = AttentionFusion(
                config.state_embedding_dim,
                config.constraint_embedding_dim,
                fusion_dim
            )
        elif config.fusion_method == "film":
            fusion_dim = config.state_embedding_dim
            self.fusion = FiLMLayer(
                config.state_embedding_dim,
                config.constraint_embedding_dim
            )
        
        # 价值网络
        layers = []
        input_dim = fusion_dim
        
        for hidden_dim in config.value_config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if config.value_config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif config.value_config.layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(config.get_activation_fn(config.value_config.activation))
            
            if config.value_config.dropout_rate > 0:
                layers.append(nn.Dropout(config.value_config.dropout_rate))
            
            input_dim = hidden_dim
        
        # 输出层 - 单一价值
        layers.append(nn.Linear(input_dim, 1))
        
        self.value_head = nn.Sequential(*layers)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化网络权重"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, states: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            states: [batch_size, state_dim]
            constraints: [batch_size, constraint_dim]
            
        Returns:
            values: [batch_size, 1] 状态价值
        """
        # 嵌入
        state_emb = self.state_embedding(states)
        constraint_emb = self.constraint_embedding(constraints)
        
        # 融合
        if self.config.fusion_method == "concat":
            fused = torch.cat([state_emb, constraint_emb], dim=-1)
        elif self.config.fusion_method == "attention":
            fused = self.fusion(state_emb, constraint_emb)
        elif self.config.fusion_method == "film":
            fused = self.fusion(state_emb, constraint_emb)
        
        # 价值输出
        values = self.value_head(fused)
        return values