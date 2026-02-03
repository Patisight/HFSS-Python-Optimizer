"""
智能体配置

定义参数化强化学习智能体的网络架构和训练参数
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import torch.nn as nn

@dataclass
class NetworkConfig:
    """神经网络配置"""
    hidden_dims: List[int] = None
    activation: str = "relu"
    dropout_rate: float = 0.0
    batch_norm: bool = False
    layer_norm: bool = True
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128]

@dataclass
class AgentConfig:
    """
    参数化强化学习智能体配置
    
    定义条件策略网络π(a|s,c)的架构和训练超参数
    """
    
    # 环境参数
    state_dim: int = 64 * 64  # 像素状态维度 (64x64网格)
    action_dim: int = 64 * 64  # 动作维度 (每个像素开/关)
    constraint_dim: int = 3    # 约束向量维度 [freq_low, freq_high, target_s11]
    
    # 网络架构
    policy_config: NetworkConfig = None
    value_config: NetworkConfig = None
    
    # 条件策略网络参数
    constraint_embedding_dim: int = 64  # 约束嵌入维度
    state_embedding_dim: int = 128      # 状态嵌入维度
    fusion_method: str = "concat"       # 融合方法: "concat", "attention", "film"
    
    # PPO训练参数
    learning_rate: float = 3e-4
    value_learning_rate: float = 1e-3
    gamma: float = 0.99           # 折扣因子
    gae_lambda: float = 0.95      # GAE参数
    clip_epsilon: float = 0.2     # PPO裁剪参数
    entropy_coef: float = 0.01    # 熵正则化系数
    value_coef: float = 0.5       # 价值损失系数
    max_grad_norm: float = 0.5    # 梯度裁剪
    
    # 训练配置
    batch_size: int = 64
    mini_batch_size: int = 16
    ppo_epochs: int = 4
    buffer_size: int = 2048
    
    # 泛化训练参数
    constraint_curriculum: bool = True    # 是否使用约束课程学习
    meta_learning: bool = False          # 是否启用元学习
    adaptation_steps: int = 5            # 快速适应步数
    adaptation_lr: float = 1e-3          # 适应学习率
    
    # 正则化参数
    l2_reg: float = 1e-4         # L2正则化
    constraint_consistency: float = 0.1  # 约束一致性损失权重
    
    # 设备配置
    device: str = "cuda"         # 训练设备
    num_workers: int = 4         # 数据加载器工作进程数
    
    def __post_init__(self):
        """初始化默认网络配置"""
        if self.policy_config is None:
            self.policy_config = NetworkConfig(
                hidden_dims=[512, 256, 128],
                activation="relu",
                dropout_rate=0.1,
                layer_norm=True
            )
        
        if self.value_config is None:
            self.value_config = NetworkConfig(
                hidden_dims=[256, 128, 64],
                activation="relu", 
                dropout_rate=0.1,
                layer_norm=True
            )
    
    def get_activation_fn(self, activation: str) -> nn.Module:
        """获取激活函数"""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU()
        }
        
        if activation.lower() not in activations:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        return activations[activation.lower()]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        def convert_value(value):
            """转换值为JSON可序列化格式"""
            import numpy as np
            if isinstance(value, (np.integer, np.int64, np.int32)):
                return int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                return float(value)
            elif hasattr(value, 'to_dict'):
                return value.to_dict()
            elif isinstance(value, list):
                return [convert_value(v) for v in value]
            return value
        
        return {
            'state_dim': convert_value(self.state_dim),
            'action_dim': convert_value(self.action_dim),
            'constraint_dim': convert_value(self.constraint_dim),
            'policy_config': {
                'hidden_dims': self.policy_config.hidden_dims,
                'activation': self.policy_config.activation,
                'dropout_rate': self.policy_config.dropout_rate,
                'batch_norm': self.policy_config.batch_norm,
                'layer_norm': self.policy_config.layer_norm
            } if self.policy_config else None,
            'value_config': {
                'hidden_dims': self.value_config.hidden_dims,
                'activation': self.value_config.activation,
                'dropout_rate': self.value_config.dropout_rate,
                'batch_norm': self.value_config.batch_norm,
                'layer_norm': self.value_config.layer_norm
            } if self.value_config else None,
            'constraint_embedding_dim': convert_value(self.constraint_embedding_dim),
            'state_embedding_dim': convert_value(self.state_embedding_dim),
            'fusion_method': self.fusion_method,
            'learning_rate': convert_value(self.learning_rate),
            'value_learning_rate': convert_value(self.value_learning_rate),
            'gamma': convert_value(self.gamma),
            'gae_lambda': convert_value(self.gae_lambda),
            'clip_epsilon': convert_value(self.clip_epsilon),
            'entropy_coef': convert_value(self.entropy_coef),
            'value_coef': convert_value(self.value_coef),
            'max_grad_norm': convert_value(self.max_grad_norm),
            'batch_size': convert_value(self.batch_size),
            'mini_batch_size': convert_value(self.mini_batch_size),
            'ppo_epochs': convert_value(self.ppo_epochs),
            'buffer_size': convert_value(self.buffer_size),
            'constraint_curriculum': self.constraint_curriculum,
            'meta_learning': self.meta_learning,
            'adaptation_steps': convert_value(self.adaptation_steps),
            'adaptation_lr': convert_value(self.adaptation_lr),
            'l2_reg': convert_value(self.l2_reg),
            'constraint_consistency': convert_value(self.constraint_consistency),
            'device': self.device,
            'num_workers': convert_value(self.num_workers)
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """验证配置参数的合理性"""
        try:
            assert self.state_dim > 0, "状态维度必须为正数"
            assert self.action_dim > 0, "动作维度必须为正数"
            assert self.constraint_dim > 0, "约束维度必须为正数"
            
            assert 0 < self.learning_rate < 1, "学习率必须在(0,1)范围内"
            assert 0 < self.gamma <= 1, "折扣因子必须在(0,1]范围内"
            assert 0 <= self.gae_lambda <= 1, "GAE参数必须在[0,1]范围内"
            assert 0 < self.clip_epsilon < 1, "PPO裁剪参数必须在(0,1)范围内"
            
            assert self.batch_size > 0, "批次大小必须为正数"
            assert self.mini_batch_size > 0, "小批次大小必须为正数"
            assert self.mini_batch_size <= self.batch_size, "小批次大小不能超过批次大小"
            
            assert self.fusion_method in ["concat", "attention", "film"], "不支持的融合方法"
            
            return True
            
        except AssertionError as e:
            raise ValueError(f"配置验证失败: {e}")
        except Exception as e:
            raise ValueError(f"配置验证出错: {e}")