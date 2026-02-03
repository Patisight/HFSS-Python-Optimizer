"""
课程学习调度器

实现自适应课程学习，根据智能体性能动态调整约束难度和采样策略
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import logging

from ..config.constraint_config import ConstraintConfig
from ..config.constraint_sampler import ConstraintSampler, SamplingStrategy, SamplingConfig

logger = logging.getLogger(__name__)

@dataclass
class CurriculumStage:
    """课程学习阶段"""
    stage_id: int
    difficulty: float  # 0.0 - 1.0
    episodes: int
    sampling_strategy: SamplingStrategy
    constraint_ranges: Dict[str, Tuple[float, float]]
    performance_threshold: float = 0.7  # 进入下一阶段的性能阈值

class CurriculumScheduler:
    """
    课程学习调度器
    
    根据智能体性能自适应调整训练难度和约束采样策略
    """
    
    def __init__(
        self,
        stages: int = 5,
        total_episodes: int = 10000,
        performance_window: int = 100,
        min_stage_episodes: int = 500,
        adaptive_threshold: float = 0.1,
        difficulty_progression: str = "adaptive"
    ):
        """
        初始化课程调度器
        
        Args:
            stages: 课程阶段数
            total_episodes: 总训练回合数
            performance_window: 性能评估窗口
            min_stage_episodes: 每阶段最小回合数
            adaptive_threshold: 自适应调整阈值
            difficulty_progression: 难度递增方式
        """
        self.stages = stages
        self.total_episodes = total_episodes
        self.performance_window = performance_window
        self.min_stage_episodes = min_stage_episodes
        self.adaptive_threshold = adaptive_threshold
        self.difficulty_progression = difficulty_progression
        
        # 当前状态
        self.current_stage = 0
        self.current_episode = 0
        self.stage_episodes = 0
        
        # 性能监控
        self.performance_history = deque(maxlen=performance_window)
        self.stage_performance = []
        self.best_performance = -float('inf')
        
        # 课程阶段
        self.curriculum_stages = self._initialize_stages()
        
        # 统计信息
        self.stage_transitions = []
        self.performance_stats = {}
        
        logger.info(f"初始化课程调度器: {stages}阶段, {total_episodes}回合")
    
    def _initialize_stages(self) -> List[CurriculumStage]:
        """初始化课程阶段"""
        stages = []
        episodes_per_stage = max(
            self.min_stage_episodes,
            self.total_episodes // self.stages
        )
        
        for i in range(self.stages):
            progress = i / max(1, self.stages - 1)
            
            # 计算难度
            if self.difficulty_progression == "linear":
                difficulty = progress
            elif self.difficulty_progression == "exponential":
                difficulty = progress ** 2
            elif self.difficulty_progression == "sqrt":
                difficulty = np.sqrt(progress)
            else:  # adaptive
                difficulty = progress
            
            # 选择采样策略
            if i == 0:
                strategy = SamplingStrategy.UNIFORM
            elif i < self.stages // 2:
                strategy = SamplingStrategy.GAUSSIAN
            elif i < self.stages * 3 // 4:
                strategy = SamplingStrategy.CURRICULUM
            else:
                strategy = SamplingStrategy.DIVERSITY
            
            # 定义约束范围（示例）
            constraint_ranges = {
                'frequency_range': (
                    1.0 + difficulty * 9.0,  # 1-10 GHz
                    2.0 + difficulty * 18.0  # 2-20 GHz
                ),
                's11_target': (
                    -10.0 - difficulty * 30.0,  # -10 to -40 dB
                    -5.0 - difficulty * 15.0    # -5 to -20 dB
                ),
                'bandwidth_ratio': (
                    0.05 + difficulty * 0.15,   # 5% to 20%
                    0.1 + difficulty * 0.4      # 10% to 50%
                )
            }
            
            stage = CurriculumStage(
                stage_id=i,
                difficulty=difficulty,
                episodes=episodes_per_stage,
                sampling_strategy=strategy,
                constraint_ranges=constraint_ranges,
                performance_threshold=0.6 + difficulty * 0.2
            )
            
            stages.append(stage)
            logger.debug(f"阶段 {i}: 难度={difficulty:.2f}, 策略={strategy.value}")
        
        return stages
    
    def update(self, episode: int, reward: float, success: bool) -> bool:
        """
        更新课程状态
        
        Args:
            episode: 当前回合数
            reward: 回合奖励
            success: 是否成功
            
        Returns:
            是否进入下一阶段
        """
        self.current_episode = episode
        self.stage_episodes += 1
        
        # 更新性能历史
        self.performance_history.append(reward)
        
        # 检查是否需要进入下一阶段
        stage_changed = False
        
        if self._should_advance_stage():
            if self.current_stage < len(self.curriculum_stages) - 1:
                self._advance_stage()
                stage_changed = True
        
        # 更新统计信息
        self._update_stats(reward, success)
        
        return stage_changed
    
    def _should_advance_stage(self) -> bool:
        """判断是否应该进入下一阶段"""
        current_stage_config = self.curriculum_stages[self.current_stage]
        
        # 检查最小回合数
        if self.stage_episodes < self.min_stage_episodes:
            return False
        
        # 检查性能阈值
        if len(self.performance_history) >= self.performance_window:
            avg_performance = np.mean(list(self.performance_history))
            
            if self.difficulty_progression == "adaptive":
                # 自适应调整阈值
                threshold = self._calculate_adaptive_threshold()
            else:
                threshold = current_stage_config.performance_threshold
            
            if avg_performance >= threshold:
                logger.info(
                    f"阶段 {self.current_stage} 性能达标: "
                    f"{avg_performance:.3f} >= {threshold:.3f}"
                )
                return True
        
        # 检查最大回合数
        if self.stage_episodes >= current_stage_config.episodes * 2:
            logger.warning(
                f"阶段 {self.current_stage} 超时，强制进入下一阶段"
            )
            return True
        
        return False
    
    def _calculate_adaptive_threshold(self) -> float:
        """计算自适应性能阈值"""
        if not self.stage_performance:
            return 0.6
        
        # 基于历史性能动态调整
        recent_performance = np.mean(self.stage_performance[-3:])
        base_threshold = 0.6 + self.current_stage * 0.05
        
        # 如果性能提升缓慢，降低阈值
        if len(self.stage_performance) >= 2:
            improvement = self.stage_performance[-1] - self.stage_performance[-2]
            if improvement < self.adaptive_threshold:
                base_threshold *= 0.9
        
        return min(base_threshold, 0.9)
    
    def _advance_stage(self):
        """进入下一阶段"""
        # 记录当前阶段性能
        if self.performance_history:
            stage_perf = np.mean(list(self.performance_history))
            self.stage_performance.append(stage_perf)
        
        # 记录阶段转换
        self.stage_transitions.append({
            'from_stage': self.current_stage,
            'to_stage': self.current_stage + 1,
            'episode': self.current_episode,
            'stage_episodes': self.stage_episodes,
            'performance': stage_perf if self.performance_history else 0.0
        })
        
        # 更新状态
        self.current_stage += 1
        self.stage_episodes = 0
        self.performance_history.clear()
        
        logger.info(
            f"进入阶段 {self.current_stage}, "
            f"难度: {self.get_current_difficulty():.2f}"
        )
    
    def _update_stats(self, reward: float, success: bool):
        """更新统计信息"""
        stage_key = f"stage_{self.current_stage}"
        
        if stage_key not in self.performance_stats:
            self.performance_stats[stage_key] = {
                'rewards': [],
                'successes': [],
                'episodes': 0
            }
        
        stats = self.performance_stats[stage_key]
        stats['rewards'].append(reward)
        stats['successes'].append(success)
        stats['episodes'] += 1
        
        # 更新最佳性能
        if reward > self.best_performance:
            self.best_performance = reward
    
    def get_current_stage(self) -> CurriculumStage:
        """获取当前阶段配置"""
        return self.curriculum_stages[self.current_stage]
    
    def get_current_difficulty(self) -> float:
        """获取当前难度"""
        return self.curriculum_stages[self.current_stage].difficulty
    
    def get_current_sampling_config(self) -> SamplingConfig:
        """获取当前采样配置"""
        stage = self.get_current_stage()
        
        return SamplingConfig(
            strategy=stage.sampling_strategy,
            num_samples=50 + int(stage.difficulty * 50),  # 50-100 samples
            gaussian_std=0.1 + stage.difficulty * 0.2,    # 0.1-0.3 std
            curriculum_start_ratio=stage.difficulty,
            curriculum_end_ratio=min(1.0, stage.difficulty + 0.3),
            diversity_clusters=max(3, int(stage.difficulty * 10))
        )
    
    def sample_constraints(
        self,
        sampler: ConstraintSampler,
        num_constraints: int = 1
    ) -> List[ConstraintConfig]:
        """
        根据当前阶段采样约束
        
        Args:
            sampler: 约束采样器
            num_constraints: 采样约束数量
            
        Returns:
            采样的约束列表
        """
        stage = self.get_current_stage()
        sampling_config = self.get_current_sampling_config()
        
        # 更新采样器配置
        sampler.config = sampling_config
        
        # 采样约束
        constraints = sampler.sample_constraints(num_constraints)
        
        # 根据阶段调整约束参数
        adjusted_constraints = []
        for constraint in constraints:
            adjusted = self._adjust_constraint_for_stage(constraint, stage)
            adjusted_constraints.append(adjusted)
        
        return adjusted_constraints
    
    def _adjust_constraint_for_stage(
        self,
        constraint: ConstraintConfig,
        stage: CurriculumStage
    ) -> ConstraintConfig:
        """根据阶段调整约束参数"""
        # 创建约束副本
        adjusted = ConstraintConfig(
            freq_low=constraint.freq_low,
            freq_high=constraint.freq_high,
            target_s11=constraint.target_s11,
            tolerance=constraint.tolerance,
            weight=constraint.weight,
            name=f"{constraint.name}_stage_{stage.stage_id}"
        )
        
        # 根据阶段难度调整参数
        difficulty = stage.difficulty
        
        # 调整频率范围 - 难度越高，频带越窄
        if 'frequency_range' in stage.constraint_ranges:
            min_freq, max_freq = stage.constraint_ranges['frequency_range']
            freq_span = max_freq - min_freq
            center_freq = (constraint.freq_low + constraint.freq_high) / 2
            
            # 根据难度调整带宽
            original_bandwidth = constraint.freq_high - constraint.freq_low
            min_bandwidth = 0.1e9  # 最小100MHz
            adjusted_bandwidth = max(min_bandwidth, original_bandwidth * (1.0 - difficulty * 0.5))
            
            # 确保调整后的频率在允许范围内
            half_bandwidth = adjusted_bandwidth / 2
            adjusted.freq_low = max(min_freq, center_freq - half_bandwidth)
            adjusted.freq_high = min(max_freq, center_freq + half_bandwidth)
        
        # 调整S11目标 - 难度越高，要求越严格
        if 's11_target' in stage.constraint_ranges:
            min_s11, max_s11 = stage.constraint_ranges['s11_target']
            # 线性插值调整目标S11
            adjusted.target_s11 = max_s11 + difficulty * (min_s11 - max_s11)
        
        # 调整容差 - 难度越高，容差越小
        if 'tolerance' in stage.constraint_ranges:
            min_tol, max_tol = stage.constraint_ranges['tolerance']
            adjusted.tolerance = max_tol - difficulty * (max_tol - min_tol)
        
        return adjusted
    
    def get_progress(self) -> Dict[str, Any]:
        """获取训练进度信息"""
        current_stage = self.get_current_stage()
        
        return {
            'current_stage': self.current_stage,
            'total_stages': len(self.curriculum_stages),
            'stage_progress': self.stage_episodes / current_stage.episodes,
            'overall_progress': self.current_episode / self.total_episodes,
            'current_difficulty': current_stage.difficulty,
            'current_strategy': current_stage.sampling_strategy.value,
            'recent_performance': (
                np.mean(list(self.performance_history))
                if self.performance_history else 0.0
            ),
            'best_performance': self.best_performance,
            'stage_transitions': len(self.stage_transitions)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取详细统计信息"""
        stats = {
            'curriculum_progress': self.get_progress(),
            'stage_performance': self.stage_performance,
            'stage_transitions': self.stage_transitions,
            'performance_stats': {}
        }
        
        # 计算每阶段统计
        for stage_key, stage_stats in self.performance_stats.items():
            if stage_stats['episodes'] > 0:
                stats['performance_stats'][stage_key] = {
                    'episodes': stage_stats['episodes'],
                    'avg_reward': np.mean(stage_stats['rewards']),
                    'std_reward': np.std(stage_stats['rewards']),
                    'success_rate': np.mean(stage_stats['successes']),
                    'max_reward': np.max(stage_stats['rewards']),
                    'min_reward': np.min(stage_stats['rewards'])
                }
        
        return stats
    
    def reset(self):
        """重置课程调度器"""
        self.current_stage = 0
        self.current_episode = 0
        self.stage_episodes = 0
        self.performance_history.clear()
        self.stage_performance.clear()
        self.best_performance = -float('inf')
        self.stage_transitions.clear()
        self.performance_stats.clear()
        
        logger.info("课程调度器已重置")
    
    def is_completed(self) -> bool:
        """检查课程是否完成"""
        return (
            self.current_stage >= len(self.curriculum_stages) - 1 and
            self.current_episode >= self.total_episodes
        )