"""
约束采样器 - 支持多样化约束生成和采样策略
用于训练数据生成和泛化测试

核心功能:
1. Latin Hypercube采样: 确保约束空间的均匀覆盖
2. 分层采样: 支持不同难度级别的约束生成
3. 物理约束: 确保生成的约束符合电磁物理规律
4. 自适应采样: 根据训练进度调整采样策略
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass, asdict
from scipy.stats import qmc
import json
import logging
from enum import Enum

# 导入约束配置
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constraint.types import ConstraintConfig
from constraint.constraint_manager import ConstraintGroup

class SamplingStrategy(Enum):
    """采样策略枚举"""
    RANDOM = "random"
    LATIN_HYPERCUBE = "latin_hypercube"
    STRATIFIED = "stratified"
    ADAPTIVE = "adaptive"

class DifficultyLevel(Enum):
    """难度级别枚举"""
    EASY = "easy"      # 宽带宽、宽松目标
    MEDIUM = "medium"  # 中等带宽、中等目标
    HARD = "hard"      # 窄带宽、严格目标
    EXTREME = "extreme" # 极窄带宽、极严格目标

@dataclass
class SamplingBounds:
    """采样边界配置"""
    freq_low_min: float = 4e9    # 最低频率下限 (Hz)
    freq_low_max: float = 5e9    # 最高频率下限 (Hz)
    bandwidth_min: float = 0.2e9 # 最小带宽 (Hz)
    bandwidth_max: float = 1e9   # 最大带宽 (Hz)
    target_s11_min: float = -30.0 # 最严格目标 (dB)
    target_s11_max: float = -5.0  # 最宽松目标 (dB)
    tolerance_min: float = 0.5    # 最小容差 (dB)
    tolerance_max: float = 5.0    # 最大容差 (dB)

class ConstraintSampler:
    """
    约束采样器
    
    支持多种采样策略，确保训练数据的多样性和覆盖性
    """
    
    def __init__(self, 
                 bounds: Optional[SamplingBounds] = None,
                 strategy: SamplingStrategy = SamplingStrategy.LATIN_HYPERCUBE,
                 seed: Optional[int] = None):
        """
        初始化约束采样器
        
        Args:
            bounds: 采样边界配置
            strategy: 采样策略
            seed: 随机种子
        """
        self.bounds = bounds or SamplingBounds()
        self.strategy = strategy
        self.rng = np.random.RandomState(seed)
        
        # 采样历史
        self.sampling_history: List[ConstraintConfig] = []
        
        # 日志配置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def sample_constraints(self, 
                          n_samples: int,
                          difficulty_distribution: Optional[Dict[DifficultyLevel, float]] = None,
                          strategy_override: Optional[SamplingStrategy] = None) -> List[ConstraintConfig]:
        """
        采样约束配置
        
        Args:
            n_samples: 采样数量
            difficulty_distribution: 难度分布 {难度: 比例}
            strategy_override: 临时覆盖采样策略
            
        Returns:
            约束配置列表
        """
        strategy = strategy_override or self.strategy
        
        # 默认难度分布
        if difficulty_distribution is None:
            difficulty_distribution = {
                DifficultyLevel.EASY: 0.3,
                DifficultyLevel.MEDIUM: 0.4,
                DifficultyLevel.HARD: 0.25,
                DifficultyLevel.EXTREME: 0.05
            }
            
        # 根据难度分布分配样本数量
        difficulty_counts = self._allocate_samples_by_difficulty(n_samples, difficulty_distribution)
        
        constraints = []
        
        for difficulty, count in difficulty_counts.items():
            if count > 0:
                difficulty_constraints = self._sample_by_strategy(count, strategy, difficulty)
                constraints.extend(difficulty_constraints)
                
        # 打乱顺序
        self.rng.shuffle(constraints)
        
        # 记录采样历史
        self.sampling_history.extend(constraints)
        
        self.logger.info(f"采样完成: {n_samples}个约束, 策略={strategy.value}")
        self._log_sampling_statistics(constraints)
        
        return constraints
        
    def _allocate_samples_by_difficulty(self, 
                                      n_samples: int, 
                                      distribution: Dict[DifficultyLevel, float]) -> Dict[DifficultyLevel, int]:
        """根据难度分布分配样本数量"""
        counts = {}
        remaining = n_samples
        
        # 按比例分配
        for difficulty, ratio in distribution.items():
            count = int(n_samples * ratio)
            counts[difficulty] = count
            remaining -= count
            
        # 将剩余样本分配给第一个难度级别
        if remaining > 0:
            first_difficulty = list(distribution.keys())[0]
            counts[first_difficulty] += remaining
            
        return counts
        
    def _sample_by_strategy(self, 
                           n_samples: int, 
                           strategy: SamplingStrategy,
                           difficulty: DifficultyLevel) -> List[ConstraintConfig]:
        """根据策略和难度采样"""
        if strategy == SamplingStrategy.LATIN_HYPERCUBE:
            return self._latin_hypercube_sampling(n_samples, difficulty)
        elif strategy == SamplingStrategy.RANDOM:
            return self._random_sampling(n_samples, difficulty)
        elif strategy == SamplingStrategy.STRATIFIED:
            return self._stratified_sampling(n_samples, difficulty)
        elif strategy == SamplingStrategy.ADAPTIVE:
            return self._adaptive_sampling(n_samples, difficulty)
        else:
            raise ValueError(f"未知采样策略: {strategy}")
            
    def _latin_hypercube_sampling(self, n_samples: int, difficulty: DifficultyLevel) -> List[ConstraintConfig]:
        """Latin Hypercube采样"""
        # 获取难度相关的边界
        bounds = self._get_difficulty_bounds(difficulty)
        
        # 创建LHS采样器
        sampler = qmc.LatinHypercube(d=4, seed=self.rng.randint(0, 2**31))
        samples = sampler.random(n_samples)
        
        constraints = []
        
        for sample in samples:
            # 映射到实际参数空间
            freq_low = bounds['freq_low_min'] + sample[0] * (bounds['freq_low_max'] - bounds['freq_low_min'])
            bandwidth = bounds['bandwidth_min'] + sample[1] * (bounds['bandwidth_max'] - bounds['bandwidth_min'])
            freq_high = freq_low + bandwidth
            
            target_s11 = bounds['target_s11_min'] + sample[2] * (bounds['target_s11_max'] - bounds['target_s11_min'])
            tolerance = bounds['tolerance_min'] + sample[3] * (bounds['tolerance_max'] - bounds['tolerance_min'])
            
            # 物理约束检查
            freq_low, freq_high, target_s11, tolerance = self._apply_physical_constraints(
                freq_low, freq_high, target_s11, tolerance
            )
            
            constraint = ConstraintConfig(
                freq_low=freq_low,
                freq_high=freq_high,
                target_s11=target_s11,
                tolerance=tolerance,
                name=f"LHS_{difficulty.value}_{len(constraints)}"
            )
            
            constraints.append(constraint)
            
        return constraints
        
    def _random_sampling(self, n_samples: int, difficulty: DifficultyLevel) -> List[ConstraintConfig]:
        """随机采样"""
        bounds = self._get_difficulty_bounds(difficulty)
        constraints = []
        
        for i in range(n_samples):
            freq_low = self.rng.uniform(bounds['freq_low_min'], bounds['freq_low_max'])
            bandwidth = self.rng.uniform(bounds['bandwidth_min'], bounds['bandwidth_max'])
            freq_high = freq_low + bandwidth
            
            target_s11 = self.rng.uniform(bounds['target_s11_min'], bounds['target_s11_max'])
            tolerance = self.rng.uniform(bounds['tolerance_min'], bounds['tolerance_max'])
            
            # 物理约束检查
            freq_low, freq_high, target_s11, tolerance = self._apply_physical_constraints(
                freq_low, freq_high, target_s11, tolerance
            )
            
            constraint = ConstraintConfig(
                freq_low=freq_low,
                freq_high=freq_high,
                target_s11=target_s11,
                tolerance=tolerance,
                name=f"Random_{difficulty.value}_{i}"
            )
            
            constraints.append(constraint)
            
        return constraints
        
    def _stratified_sampling(self, n_samples: int, difficulty: DifficultyLevel) -> List[ConstraintConfig]:
        """分层采样"""
        # 将参数空间分层
        n_strata = int(np.ceil(np.sqrt(n_samples)))
        samples_per_stratum = n_samples // (n_strata ** 2)
        
        bounds = self._get_difficulty_bounds(difficulty)
        constraints = []
        
        # 频率分层
        freq_strata = np.linspace(bounds['freq_low_min'], bounds['freq_low_max'], n_strata + 1)
        # 带宽分层
        bw_strata = np.linspace(bounds['bandwidth_min'], bounds['bandwidth_max'], n_strata + 1)
        
        for i in range(n_strata):
            for j in range(n_strata):
                for k in range(samples_per_stratum):
                    # 在每个层内随机采样
                    freq_low = self.rng.uniform(freq_strata[i], freq_strata[i+1])
                    bandwidth = self.rng.uniform(bw_strata[j], bw_strata[j+1])
                    freq_high = freq_low + bandwidth
                    
                    target_s11 = self.rng.uniform(bounds['target_s11_min'], bounds['target_s11_max'])
                    tolerance = self.rng.uniform(bounds['tolerance_min'], bounds['tolerance_max'])
                    
                    # 物理约束检查
                    freq_low, freq_high, target_s11, tolerance = self._apply_physical_constraints(
                        freq_low, freq_high, target_s11, tolerance
                    )
                    
                    constraint = ConstraintConfig(
                        freq_low=freq_low,
                        freq_high=freq_high,
                        target_s11=target_s11,
                        tolerance=tolerance,
                        name=f"Stratified_{difficulty.value}_{i}_{j}_{k}"
                    )
                    
                    constraints.append(constraint)
                    
        return constraints[:n_samples]  # 确保返回正确数量
        
    def _adaptive_sampling(self, n_samples: int, difficulty: DifficultyLevel) -> List[ConstraintConfig]:
        """自适应采样（根据历史性能调整）"""
        # 如果没有历史数据，使用Latin Hypercube
        if len(self.sampling_history) < 10:
            return self._latin_hypercube_sampling(n_samples, difficulty)
            
        # 分析历史约束的性能分布
        # 这里简化为基于历史约束的变异采样
        base_constraints = self.rng.choice(self.sampling_history, 
                                         size=min(n_samples//2, len(self.sampling_history)), 
                                         replace=False)
        
        constraints = []
        bounds = self._get_difficulty_bounds(difficulty)
        
        # 基于历史约束的变异
        for base in base_constraints:
            # 添加噪声变异
            noise_scale = 0.1  # 10%的变异
            
            freq_range = base.freq_high - base.freq_low
            freq_center = (base.freq_low + base.freq_high) / 2
            
            new_center = freq_center * (1 + self.rng.normal(0, noise_scale))
            new_range = freq_range * (1 + self.rng.normal(0, noise_scale))
            
            new_freq_low = new_center - new_range / 2
            new_freq_high = new_center + new_range / 2
            
            new_target_s11 = base.target_s11 * (1 + self.rng.normal(0, noise_scale))
            new_tolerance = base.tolerance * (1 + self.rng.normal(0, noise_scale))
            
            # 物理约束检查
            new_freq_low, new_freq_high, new_target_s11, new_tolerance = self._apply_physical_constraints(
                new_freq_low, new_freq_high, new_target_s11, new_tolerance
            )
            
            constraint = ConstraintConfig(
                freq_low=new_freq_low,
                freq_high=new_freq_high,
                target_s11=new_target_s11,
                tolerance=new_tolerance,
                name=f"Adaptive_{difficulty.value}_{len(constraints)}"
            )
            
            constraints.append(constraint)
            
        # 剩余样本用Latin Hypercube填充
        remaining = n_samples - len(constraints)
        if remaining > 0:
            additional = self._latin_hypercube_sampling(remaining, difficulty)
            constraints.extend(additional)
            
        return constraints
        
    def _get_difficulty_bounds(self, difficulty: DifficultyLevel) -> Dict[str, float]:
        """根据难度级别获取采样边界"""
        base_bounds = asdict(self.bounds)
        
        if difficulty == DifficultyLevel.EASY:
            # 宽带宽、宽松目标
            return {
                'freq_low_min': base_bounds['freq_low_min'],
                'freq_low_max': base_bounds['freq_low_max'] * 0.7,
                'bandwidth_min': base_bounds['bandwidth_max'] * 0.5,
                'bandwidth_max': base_bounds['bandwidth_max'],
                'target_s11_min': base_bounds['target_s11_max'] * 0.7,
                'target_s11_max': base_bounds['target_s11_max'],
                'tolerance_min': base_bounds['tolerance_max'] * 0.6,
                'tolerance_max': base_bounds['tolerance_max']
            }
        elif difficulty == DifficultyLevel.MEDIUM:
            # 中等设置
            return {
                'freq_low_min': base_bounds['freq_low_min'],
                'freq_low_max': base_bounds['freq_low_max'] * 0.8,
                'bandwidth_min': base_bounds['bandwidth_min'] * 2,
                'bandwidth_max': base_bounds['bandwidth_max'] * 0.7,
                'target_s11_min': (base_bounds['target_s11_min'] + base_bounds['target_s11_max']) / 2,
                'target_s11_max': base_bounds['target_s11_max'] * 0.8,
                'tolerance_min': base_bounds['tolerance_min'] * 2,
                'tolerance_max': base_bounds['tolerance_max'] * 0.7
            }
        elif difficulty == DifficultyLevel.HARD:
            # 窄带宽、严格目标
            return {
                'freq_low_min': base_bounds['freq_low_min'],
                'freq_low_max': base_bounds['freq_low_max'],
                'bandwidth_min': base_bounds['bandwidth_min'],
                'bandwidth_max': base_bounds['bandwidth_max'] * 0.4,
                'target_s11_min': base_bounds['target_s11_min'] * 0.8,
                'target_s11_max': (base_bounds['target_s11_min'] + base_bounds['target_s11_max']) / 2,
                'tolerance_min': base_bounds['tolerance_min'],
                'tolerance_max': base_bounds['tolerance_max'] * 0.4
            }
        else:  # EXTREME
            # 极端设置
            return {
                'freq_low_min': base_bounds['freq_low_min'],
                'freq_low_max': base_bounds['freq_low_max'],
                'bandwidth_min': base_bounds['bandwidth_min'],
                'bandwidth_max': base_bounds['bandwidth_min'] * 3,
                'target_s11_min': base_bounds['target_s11_min'],
                'target_s11_max': base_bounds['target_s11_min'] * 0.7,
                'tolerance_min': base_bounds['tolerance_min'],
                'tolerance_max': base_bounds['tolerance_min'] * 2
            }
            
    def _apply_physical_constraints(self, 
                                  freq_low: float, 
                                  freq_high: float, 
                                  target_s11: float, 
                                  tolerance: float) -> Tuple[float, float, float, float]:
        """应用物理约束检查和修正"""
        # 1. 频率约束
        freq_low = max(freq_low, self.bounds.freq_low_min)
        freq_high = min(freq_high, 6e9)  # 最高6GHz
        
        # 确保带宽合理
        bandwidth = freq_high - freq_low
        if bandwidth < self.bounds.bandwidth_min:
            freq_high = freq_low + self.bounds.bandwidth_min
        elif bandwidth > self.bounds.bandwidth_max:
            freq_high = freq_low + self.bounds.bandwidth_max
            
        # 2. S11约束
        target_s11 = max(target_s11, self.bounds.target_s11_min)
        target_s11 = min(target_s11, self.bounds.target_s11_max)
        
        # 3. 容差约束
        tolerance = max(tolerance, self.bounds.tolerance_min)
        tolerance = min(tolerance, self.bounds.tolerance_max)
        
        # 4. 物理合理性检查
        # 高频时S11目标应该更严格
        if freq_low > 3e9:
            target_s11 = min(target_s11, -8.0)
            
        # 窄带宽时容差应该更小
        if bandwidth < 0.5e9:
            tolerance = min(tolerance, 2.0)
            
        return freq_low, freq_high, target_s11, tolerance
        
    def _log_sampling_statistics(self, constraints: List[ConstraintConfig]):
        """记录采样统计信息"""
        if not constraints:
            return
            
        freq_lows = [c.freq_low/1e9 for c in constraints]
        freq_highs = [c.freq_high/1e9 for c in constraints]
        bandwidths = [(c.freq_high - c.freq_low)/1e9 for c in constraints]
        targets = [c.target_s11 for c in constraints]
        tolerances = [c.tolerance for c in constraints]
        
        self.logger.info("采样统计:")
        self.logger.info(f"  频率下限: {np.min(freq_lows):.1f}-{np.max(freq_lows):.1f} GHz")
        self.logger.info(f"  频率上限: {np.min(freq_highs):.1f}-{np.max(freq_highs):.1f} GHz")
        self.logger.info(f"  带宽: {np.min(bandwidths):.1f}-{np.max(bandwidths):.1f} GHz")
        self.logger.info(f"  目标S11: {np.min(targets):.1f}-{np.max(targets):.1f} dB")
        self.logger.info(f"  容差: {np.min(tolerances):.1f}-{np.max(tolerances):.1f} dB")
        
    def save_constraints(self, constraints: List[ConstraintConfig], filepath: str):
        """保存约束配置到文件"""
        data = [asdict(c) for c in constraints]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"约束配置已保存到: {filepath}")
        
    def load_constraints(self, filepath: str) -> List[ConstraintConfig]:
        """从文件加载约束配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        constraints = [ConstraintConfig(**item) for item in data]
        self.logger.info(f"从文件加载了{len(constraints)}个约束配置: {filepath}")
        return constraints
        
    def get_sampling_statistics(self) -> Dict[str, Any]:
        """获取采样统计信息"""
        if not self.sampling_history:
            return {}
            
        freq_lows = [c.freq_low for c in self.sampling_history]
        freq_highs = [c.freq_high for c in self.sampling_history]
        bandwidths = [c.freq_high - c.freq_low for c in self.sampling_history]
        targets = [c.target_s11 for c in self.sampling_history]
        tolerances = [c.tolerance for c in self.sampling_history]
        
        return {
            'total_samples': len(self.sampling_history),
            'freq_low_range': [np.min(freq_lows)/1e9, np.max(freq_lows)/1e9],
            'freq_high_range': [np.min(freq_highs)/1e9, np.max(freq_highs)/1e9],
            'bandwidth_range': [np.min(bandwidths)/1e9, np.max(bandwidths)/1e9],
            'target_s11_range': [np.min(targets), np.max(targets)],
            'tolerance_range': [np.min(tolerances), np.max(tolerances)]
        }

    def sample_constraint_groups(self,
                                 n_groups: int,
                                 bands_per_group: Tuple[int, int] = (2, 3),
                                 difficulty_distribution: Optional[Dict[DifficultyLevel, float]] = None,
                                 strategy_override: Optional[SamplingStrategy] = None,
                                 name_prefix: str = "MB",
                                 ensure_non_overlap: bool = True) -> List[ConstraintGroup]:
        """采样多频段约束组
        生成由多个ConstraintConfig组成的ConstraintGroup，用于双/三频段训练与评估。
        
        Args:
            n_groups: 约束组数量
            bands_per_group: 每组频段数量范围 (min, max)
            difficulty_distribution: 难度分布
            strategy_override: 临时覆盖采样策略
            name_prefix: 组名前缀
            ensure_non_overlap: 是否强制频段不重叠
        Returns:
            约束组列表
        """
        strategy = strategy_override or self.strategy
        if difficulty_distribution is None:
            difficulty_distribution = {
                DifficultyLevel.MEDIUM: 0.6,
                DifficultyLevel.HARD: 0.3,
                DifficultyLevel.EXTREME: 0.1
            }
        difficulty_counts = self._allocate_samples_by_difficulty(n_groups, difficulty_distribution)
        groups: List[ConstraintGroup] = []
        
        for difficulty, count in difficulty_counts.items():
            for gi in range(count):
                n_bands = self.rng.randint(bands_per_group[0], bands_per_group[1] + 1)
                constraints = self._sample_multi_band(n_bands, difficulty, strategy, ensure_non_overlap)
                group_name = f"{name_prefix}_{difficulty.value}_{n_bands}_{len(groups)}"
                groups.append(ConstraintGroup(name=group_name, constraints=constraints, priority=1.0, description=f"{n_bands}-band {difficulty.value}"))
        
        # 打乱顺序
        self.rng.shuffle(groups)
        self.logger.info(f"采样完成: {len(groups)}个约束组, 策略={strategy.value}")
        return groups
    
    def _sample_multi_band(self,
                           n_bands: int,
                           difficulty: DifficultyLevel,
                           strategy: SamplingStrategy,
                           ensure_non_overlap: bool) -> List[ConstraintConfig]:
        """生成多频段约束，尽量避免重叠并控制严格度"""
        b = self._get_difficulty_bounds(difficulty)
        constraints: List[ConstraintConfig] = []
        
        # 准备中心频率与带宽采样
        freq_max_possible = b['freq_low_max'] + b['bandwidth_max']
        gap = 0.05e9  # 50 MHz 安全间隔
        centers = self.rng.uniform(b['freq_low_min'] + gap, freq_max_possible - gap, size=n_bands)
        centers.sort()
        
        for i in range(n_bands):
            bw = self.rng.uniform(b['bandwidth_min'] * 0.6, b['bandwidth_max'] * 0.8)
            fl = max(b['freq_low_min'], centers[i] - bw/2)
            fh = min(freq_max_possible, centers[i] + bw/2)
            
            # 强制不重叠：调整下一个频段中心或扩展间隔
            if ensure_non_overlap and i > 0:
                prev = constraints[-1]
                if fl < prev.freq_high + gap:
                    shift = (prev.freq_high + gap) - fl
                    fl += shift
                    fh = fl + bw
                    if fh > freq_max_possible:
                        fh = freq_max_possible
                        fl = max(b['freq_low_min'], fh - bw)
            
            target_s11 = self.rng.uniform(b['target_s11_min'], b['target_s11_max'])
            tolerance = self.rng.uniform(b['tolerance_min'], b['tolerance_max'])
            fl, fh, target_s11, tolerance = self._apply_physical_constraints(fl, fh, target_s11, tolerance)
            
            constraints.append(ConstraintConfig(
                freq_low=fl,
                freq_high=fh,
                target_s11=target_s11,
                tolerance=tolerance,
                weight=1.0,
                name=f"Band_{i}"
            ))
        return constraints