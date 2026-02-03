"""
约束采样器

实现多样化约束采样策略，支持参数化强化学习的泛化训练
"""

from typing import List, Dict, Optional, Tuple, Callable
import numpy as np
import logging
from enum import Enum
from dataclasses import dataclass
import random

from .constraint_config import ConstraintConfig, ConstraintGroup, ConstraintManager

class SamplingStrategy(Enum):
    """约束采样策略"""
    UNIFORM = "uniform"           # 均匀采样
    GAUSSIAN = "gaussian"         # 高斯采样
    CURRICULUM = "curriculum"     # 课程学习
    ADAPTIVE = "adaptive"         # 自适应采样
    DIVERSITY = "diversity"       # 多样性采样

@dataclass
class SamplingConfig:
    """采样配置"""
    strategy: SamplingStrategy = SamplingStrategy.UNIFORM
    num_samples: int = 100
    seed: Optional[int] = None
    
    # 高斯采样参数
    gaussian_std: float = 0.1
    
    # 课程学习参数
    curriculum_start_ratio: float = 0.1
    curriculum_end_ratio: float = 1.0
    curriculum_steps: int = 1000
    
    # 自适应采样参数
    adaptive_window: int = 100
    adaptive_threshold: float = 0.8
    
    # 多样性采样参数
    diversity_clusters: int = 10
    diversity_min_distance: float = 0.1
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'strategy': self.strategy.value,
            'num_samples': self.num_samples,
            'seed': self.seed,
            'gaussian_std': self.gaussian_std,
            'curriculum_start_ratio': self.curriculum_start_ratio,
            'curriculum_end_ratio': self.curriculum_end_ratio,
            'curriculum_steps': self.curriculum_steps,
            'adaptive_window': self.adaptive_window,
            'adaptive_threshold': self.adaptive_threshold,
            'diversity_clusters': self.diversity_clusters,
            'diversity_min_distance': self.diversity_min_distance
        }

class ConstraintSampler:
    """
    约束采样器
    
    实现多种采样策略，为参数化强化学习提供多样化的约束分布
    """
    
    def __init__(self, 
                 constraint_manager: ConstraintManager,
                 config: SamplingConfig = None):
        """
        初始化约束采样器
        
        Args:
            constraint_manager: 约束管理器
            config: 采样配置
        """
        self.constraint_manager = constraint_manager
        self.config = config or SamplingConfig()
        
        # 设置随机种子
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)
        
        # 采样历史和统计
        self.sampling_history: List[ConstraintConfig] = []
        self.performance_history: List[float] = []
        self.current_step = 0
        
        # 日志配置
        self.logger = logging.getLogger(__name__)
    
    def sample_constraints(self, 
                          num_samples: Optional[int] = None,
                          strategy: Optional[SamplingStrategy] = None) -> List[ConstraintConfig]:
        """
        采样约束
        
        Args:
            num_samples: 采样数量
            strategy: 采样策略
            
        Returns:
            采样的约束列表
        """
        num_samples = num_samples or self.config.num_samples
        strategy = strategy or self.config.strategy
        
        if strategy == SamplingStrategy.UNIFORM:
            return self._uniform_sampling(num_samples)
        elif strategy == SamplingStrategy.GAUSSIAN:
            return self._gaussian_sampling(num_samples)
        elif strategy == SamplingStrategy.CURRICULUM:
            return self._curriculum_sampling(num_samples)
        elif strategy == SamplingStrategy.ADAPTIVE:
            return self._adaptive_sampling(num_samples)
        elif strategy == SamplingStrategy.DIVERSITY:
            return self._diversity_sampling(num_samples)
        else:
            raise ValueError(f"不支持的采样策略: {strategy}")
    
    def _uniform_sampling(self, num_samples: int) -> List[ConstraintConfig]:
        """均匀采样"""
        constraints = []
        freq_range = self.constraint_manager.freq_range
        s11_range = self.constraint_manager.s11_range
        
        for i in range(num_samples):
            # 随机采样频率范围
            freq_low = np.random.uniform(freq_range[0], freq_range[1] * 0.8)
            freq_high = np.random.uniform(freq_low + 0.1e9, freq_range[1])
            
            # 随机采样目标S11
            target_s11 = np.random.uniform(s11_range[0], s11_range[1])
            
            # 随机采样容差
            tolerance = np.random.uniform(0.5, 5.0)
            
            constraint = ConstraintConfig(
                freq_low=freq_low,
                freq_high=freq_high,
                target_s11=target_s11,
                tolerance=tolerance,
                name=f"uniform_sample_{i}"
            )
            
            constraints.append(constraint)
        
        self.logger.info(f"均匀采样生成 {num_samples} 个约束")
        return constraints
    
    def _gaussian_sampling(self, num_samples: int) -> List[ConstraintConfig]:
        """高斯采样 - 围绕已有约束进行采样"""
        constraints = []
        existing_constraints = self.constraint_manager.get_all_constraints()
        
        if not existing_constraints:
            # 如果没有已有约束，回退到均匀采样
            return self._uniform_sampling(num_samples)
        
        for i in range(num_samples):
            # 随机选择一个基准约束
            base_constraint = random.choice(existing_constraints)
            
            # 在基准约束周围进行高斯采样
            freq_center = base_constraint.get_center_freq()
            bandwidth = base_constraint.get_bandwidth()
            
            # 添加高斯噪声
            noise_scale = self.config.gaussian_std
            freq_noise = np.random.normal(0, freq_center * noise_scale)
            bw_noise = np.random.normal(0, bandwidth * noise_scale)
            s11_noise = np.random.normal(0, abs(base_constraint.target_s11) * noise_scale)
            
            # 生成新约束
            new_center = max(freq_center + freq_noise, self.constraint_manager.freq_range[0] + 0.5e9)
            new_bandwidth = max(bandwidth + bw_noise, 0.1e9)
            
            freq_low = max(new_center - new_bandwidth/2, self.constraint_manager.freq_range[0])
            freq_high = min(new_center + new_bandwidth/2, self.constraint_manager.freq_range[1])
            
            target_s11 = np.clip(
                base_constraint.target_s11 + s11_noise,
                self.constraint_manager.s11_range[0],
                self.constraint_manager.s11_range[1]
            )
            
            constraint = ConstraintConfig(
                freq_low=freq_low,
                freq_high=freq_high,
                target_s11=target_s11,
                tolerance=base_constraint.tolerance,
                name=f"gaussian_sample_{i}"
            )
            
            constraints.append(constraint)
        
        self.logger.info(f"高斯采样生成 {num_samples} 个约束")
        return constraints
    
    def _curriculum_sampling(self, num_samples: int) -> List[ConstraintConfig]:
        """课程学习采样 - 从简单到复杂"""
        # 计算当前课程进度
        progress = min(self.current_step / self.config.curriculum_steps, 1.0)
        difficulty_ratio = (
            self.config.curriculum_start_ratio + 
            progress * (self.config.curriculum_end_ratio - self.config.curriculum_start_ratio)
        )
        
        constraints = []
        freq_range = self.constraint_manager.freq_range
        s11_range = self.constraint_manager.s11_range
        
        for i in range(num_samples):
            # 根据难度比例调整约束复杂度
            # 简单约束：较宽频带，较松的S11要求
            # 复杂约束：较窄频带，较严格的S11要求
            
            # 频带宽度（难度越高，频带越窄）
            max_bandwidth = (freq_range[1] - freq_range[0]) * 0.8
            min_bandwidth = 0.1e9
            bandwidth = max_bandwidth - difficulty_ratio * (max_bandwidth - min_bandwidth)
            
            # 随机选择中心频率
            freq_center = np.random.uniform(
                freq_range[0] + bandwidth/2,
                freq_range[1] - bandwidth/2
            )
            
            freq_low = freq_center - bandwidth/2
            freq_high = freq_center + bandwidth/2
            
            # S11要求（难度越高，要求越严格）
            easy_s11 = -10.0  # 简单目标
            hard_s11 = -25.0  # 困难目标
            target_s11 = easy_s11 - difficulty_ratio * (easy_s11 - hard_s11)
            
            # 容差（难度越高，容差越小）
            max_tolerance = 5.0
            min_tolerance = 0.5
            tolerance = max_tolerance - difficulty_ratio * (max_tolerance - min_tolerance)
            
            constraint = ConstraintConfig(
                freq_low=freq_low,
                freq_high=freq_high,
                target_s11=target_s11,
                tolerance=tolerance,
                name=f"curriculum_sample_{i}_step_{self.current_step}"
            )
            
            constraints.append(constraint)
        
        self.current_step += 1
        self.logger.info(f"课程学习采样生成 {num_samples} 个约束，难度: {difficulty_ratio:.2f}")
        return constraints
    
    def _adaptive_sampling(self, num_samples: int) -> List[ConstraintConfig]:
        """自适应采样 - 根据性能历史调整采样"""
        if len(self.performance_history) < self.config.adaptive_window:
            # 性能历史不足，使用均匀采样
            return self._uniform_sampling(num_samples)
        
        # 计算最近窗口内的平均性能
        recent_performance = np.mean(self.performance_history[-self.config.adaptive_window:])
        
        if recent_performance > self.config.adaptive_threshold:
            # 性能良好，增加难度
            return self._curriculum_sampling(num_samples)
        else:
            # 性能不佳，降低难度或增加多样性
            return self._diversity_sampling(num_samples)
    
    def _diversity_sampling(self, num_samples: int) -> List[ConstraintConfig]:
        """多样性采样 - 确保约束分布的多样性"""
        constraints = []
        
        # 首先生成候选约束（比目标数量多）
        candidate_num = num_samples * 3
        candidates = self._uniform_sampling(candidate_num)
        
        # 将约束转换为向量表示
        candidate_vectors = np.array([c.to_vector() for c in candidates])
        
        # 使用K-means聚类选择多样化的约束
        from sklearn.cluster import KMeans
        
        n_clusters = min(self.config.diversity_clusters, num_samples)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(candidate_vectors)
        
        # 从每个聚类中选择代表性约束
        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                # 选择距离聚类中心最近的约束
                cluster_center = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(
                    candidate_vectors[cluster_indices] - cluster_center, axis=1
                )
                best_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(best_idx)
        
        # 如果选择的约束不够，随机补充
        while len(selected_indices) < num_samples:
            remaining_indices = set(range(len(candidates))) - set(selected_indices)
            if remaining_indices:
                selected_indices.append(random.choice(list(remaining_indices)))
            else:
                break
        
        # 截取到目标数量
        selected_indices = selected_indices[:num_samples]
        constraints = [candidates[i] for i in selected_indices]
        
        # 更新约束名称
        for i, constraint in enumerate(constraints):
            constraint.name = f"diversity_sample_{i}"
        
        self.logger.info(f"多样性采样生成 {num_samples} 个约束")
        return constraints
    
    def update_performance(self, constraint: ConstraintConfig, performance: float):
        """更新性能历史"""
        self.sampling_history.append(constraint)
        self.performance_history.append(performance)
        
        # 限制历史长度
        max_history = 1000
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
            self.sampling_history = self.sampling_history[-max_history:]
    
    def get_sampling_statistics(self) -> Dict:
        """获取采样统计信息"""
        if not self.performance_history:
            return {}
        
        return {
            'total_samples': len(self.performance_history),
            'mean_performance': np.mean(self.performance_history),
            'std_performance': np.std(self.performance_history),
            'best_performance': np.max(self.performance_history),
            'worst_performance': np.min(self.performance_history),
            'current_step': self.current_step
        }
    
    def reset(self):
        """重置采样器状态"""
        self.sampling_history.clear()
        self.performance_history.clear()
        self.current_step = 0
        self.logger.info("重置约束采样器状态")