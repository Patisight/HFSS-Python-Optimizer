"""
训练配置

定义参数化强化学习的训练参数、约束采样策略和课程学习配置
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json

from ..config.constraint_sampler import SamplingStrategy, SamplingConfig

@dataclass
class TrainingConfig:
    """
    泛化训练配置
    
    定义参数化强化学习的训练流程、约束采样和课程学习参数
    """
    
    # 基础训练参数
    total_episodes: int = 10000          # 总训练回合数
    max_steps_per_episode: int = 1000    # 每回合最大步数
    update_frequency: int = 10           # 更新频率（回合）
    
    # 约束采样配置
    sampling_config: SamplingConfig = None
    constraint_change_frequency: int = 50  # 约束切换频率（回合）
    
    # 课程学习参数
    curriculum_learning: bool = True      # 是否启用课程学习
    curriculum_stages: int = 5           # 课程阶段数
    stage_episodes: int = 2000           # 每阶段回合数
    difficulty_progression: str = "linear"  # 难度递增方式: "linear", "exponential", "adaptive"
    
    # 评估参数
    eval_frequency: int = 500            # 评估频率（回合）
    eval_episodes_per_constraint: int = 10  # 每约束评估回合数
    eval_constraints_num: int = 20       # 评估约束数量
    
    # 保存和日志
    save_frequency: int = 1000           # 模型保存频率（回合）
    log_frequency: int = 100             # 日志记录频率（回合）
    checkpoint_dir: str = "checkpoints"  # 检查点目录
    log_dir: str = "logs"               # 日志目录
    
    # 早停和性能监控
    early_stopping: bool = True          # 是否启用早停
    patience: int = 2000                # 早停耐心（回合）
    min_improvement: float = 0.01       # 最小改进阈值
    performance_window: int = 100       # 性能监控窗口
    
    # 多进程训练
    num_parallel_envs: int = 1          # 并行环境数
    num_workers: int = 4                # 数据加载工作进程数
    
    # 实验配置
    experiment_name: str = "generalized_training"  # 实验名称
    seed: Optional[int] = None          # 随机种子
    device: str = "cuda"                # 训练设备
    
    # 高级训练策略
    meta_learning: bool = False         # 是否启用元学习
    domain_randomization: bool = True   # 是否启用域随机化
    constraint_augmentation: bool = True  # 是否启用约束增强
    
    # 性能优化
    mixed_precision: bool = False       # 是否使用混合精度
    gradient_accumulation_steps: int = 1  # 梯度累积步数
    
    def __post_init__(self):
        """初始化默认配置"""
        if self.sampling_config is None:
            self.sampling_config = SamplingConfig(
                strategy=SamplingStrategy.CURRICULUM,
                num_samples=100,
                seed=self.seed
            )
        
        # 创建目录
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
    
    def get_stage_config(self, stage: int) -> Dict[str, Any]:
        """获取特定阶段的配置"""
        if not self.curriculum_learning:
            return {}
        
        progress = stage / max(1, self.curriculum_stages - 1)
        
        if self.difficulty_progression == "linear":
            difficulty = progress
        elif self.difficulty_progression == "exponential":
            difficulty = progress ** 2
        elif self.difficulty_progression == "adaptive":
            # 自适应难度需要根据性能动态调整
            difficulty = progress
        else:
            difficulty = progress
        
        return {
            'stage': stage,
            'progress': progress,
            'difficulty': difficulty,
            'episodes': self.stage_episodes
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, SamplingConfig):
                config_dict[key] = {
                    'strategy': value.strategy.value,
                    'num_samples': value.num_samples,
                    'seed': value.seed,
                    'gaussian_std': value.gaussian_std,
                    'curriculum_start_ratio': value.curriculum_start_ratio,
                    'curriculum_end_ratio': value.curriculum_end_ratio,
                    'curriculum_steps': value.curriculum_steps,
                    'adaptive_window': value.adaptive_window,
                    'adaptive_threshold': value.adaptive_threshold,
                    'diversity_clusters': value.diversity_clusters,
                    'diversity_min_distance': value.diversity_min_distance
                }
            else:
                config_dict[key] = value
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """从字典创建配置"""
        # 处理SamplingConfig
        if 'sampling_config' in config_dict and isinstance(config_dict['sampling_config'], dict):
            sampling_dict = config_dict['sampling_config']
            sampling_dict['strategy'] = SamplingStrategy(sampling_dict['strategy'])
            config_dict['sampling_config'] = SamplingConfig(**sampling_dict)
        
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """保存配置到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def validate(self) -> bool:
        """验证配置参数的合理性"""
        try:
            assert self.total_episodes > 0, "总回合数必须为正数"
            assert self.max_steps_per_episode > 0, "每回合最大步数必须为正数"
            assert self.update_frequency > 0, "更新频率必须为正数"
            assert self.constraint_change_frequency > 0, "约束切换频率必须为正数"
            
            if self.curriculum_learning:
                assert self.curriculum_stages > 0, "课程阶段数必须为正数"
                assert self.stage_episodes > 0, "每阶段回合数必须为正数"
                assert self.difficulty_progression in ["linear", "exponential", "adaptive"], \
                    "不支持的难度递增方式"
            
            assert self.eval_frequency > 0, "评估频率必须为正数"
            assert self.eval_episodes_per_constraint > 0, "每约束评估回合数必须为正数"
            assert self.eval_constraints_num > 0, "评估约束数量必须为正数"
            
            assert self.save_frequency > 0, "保存频率必须为正数"
            assert self.log_frequency > 0, "日志频率必须为正数"
            
            if self.early_stopping:
                assert self.patience > 0, "早停耐心必须为正数"
                assert self.min_improvement >= 0, "最小改进阈值必须非负"
                assert self.performance_window > 0, "性能监控窗口必须为正数"
            
            assert self.num_parallel_envs > 0, "并行环境数必须为正数"
            assert self.num_workers >= 0, "工作进程数必须非负"
            assert self.gradient_accumulation_steps > 0, "梯度累积步数必须为正数"
            
            return True
            
        except AssertionError as e:
            raise ValueError(f"训练配置验证失败: {e}")
        except Exception as e:
            raise ValueError(f"训练配置验证出错: {e}")
    
    def get_experiment_dir(self) -> Path:
        """获取实验目录"""
        return Path(self.checkpoint_dir) / self.experiment_name
    
    def get_log_dir(self) -> Path:
        """获取日志目录"""
        return Path(self.log_dir) / self.experiment_name