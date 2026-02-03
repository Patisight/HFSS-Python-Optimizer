"""
训练模块

实现参数化强化学习的泛化训练管道，支持多样约束采样和课程学习
"""

from .training_config import TrainingConfig
from .generalized_trainer import GeneralizedTrainer
from .curriculum_scheduler import CurriculumScheduler

__all__ = [
    'TrainingConfig',
    'GeneralizedTrainer', 
    'CurriculumScheduler'
]