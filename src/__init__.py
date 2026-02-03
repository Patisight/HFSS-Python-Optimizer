"""
泛化强化学习像素天线优化系统

本系统实现基于参数化强化学习的通用像素天线优化框架，
支持任意频段约束的零样本泛化能力。
"""

__version__ = "1.0.0"
__author__ = "Pixel Antenna RL Team"

# 核心模块导入
from .environment.parameterized_env import ParameterizedPixelAntennaEnv
from .agent.generalized_agent import GeneralizedPPOAgent
from .agent.agent_config import AgentConfig
from .config.constraint_config import ConstraintConfig, ConstraintManager
from .training.generalized_trainer import GeneralizedTrainer, TrainingConfig

__all__ = [
    "ParameterizedPixelAntennaEnv",
    "GeneralizedPPOAgent", 
    "AgentConfig",
    "ConstraintConfig",
    "ConstraintManager", 
    "GeneralizedTrainer",
    "TrainingConfig"
]