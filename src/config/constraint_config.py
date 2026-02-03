"""
约束配置系统

提供约束定义、验证、管理和采样功能，支持参数化强化学习的动态约束注入
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import logging
from enum import Enum
import json
from pathlib import Path

@dataclass
class ConstraintConfig:
    """
    通用约束配置类型
    
    定义频段约束和目标S11值，支持参数化强化学习的约束向量表示
    """
    freq_low: float      # 频段下限 (Hz)
    freq_high: float     # 频段上限 (Hz) 
    target_s11: float    # 目标S11值 (dB)
    tolerance: float = 2.0    # 容差 (dB)
    weight: float = 1.0       # 权重
    name: str = "constraint"  # 约束名称
    
    def __post_init__(self):
        """验证约束参数的合理性"""
        if self.freq_low >= self.freq_high:
            raise ValueError(f"频段下限 {self.freq_low} 必须小于上限 {self.freq_high}")
        if self.target_s11 > 0:
            raise ValueError(f"目标S11值 {self.target_s11} 必须为负值")
        if self.tolerance <= 0:
            raise ValueError(f"容差 {self.tolerance} 必须为正值")
        if self.weight <= 0:
            raise ValueError(f"权重 {self.weight} 必须为正值")
    
    def to_vector(self) -> np.ndarray:
        """转换为约束向量表示，用于参数化RL状态空间"""
        return np.array([
            self.freq_low / 1e9,    # 归一化到GHz
            self.freq_high / 1e9,   # 归一化到GHz
            self.target_s11 / 50.0  # 归一化到[-1, 0]范围
        ], dtype=np.float32)
    
    @classmethod
    def from_vector(cls, vector: np.ndarray, name: str = "constraint") -> 'ConstraintConfig':
        """从约束向量创建约束配置"""
        return cls(
            freq_low=float(vector[0] * 1e9),
            freq_high=float(vector[1] * 1e9), 
            target_s11=float(vector[2] * 50.0),
            name=name
        )
    
    def get_bandwidth(self) -> float:
        """获取带宽 (Hz)"""
        return self.freq_high - self.freq_low
    
    def get_center_freq(self) -> float:
        """获取中心频率 (Hz)"""
        return (self.freq_low + self.freq_high) / 2
    
    def is_in_band(self, freq: float) -> bool:
        """判断频率是否在约束频段内"""
        return self.freq_low <= freq <= self.freq_high

@dataclass
class ConstraintGroup:
    """约束组 - 支持多约束组合优化"""
    name: str
    constraints: List[ConstraintConfig]
    priority: float = 1.0
    description: str = ""
    
    def to_vector(self) -> np.ndarray:
        """将约束组转换为向量表示"""
        if not self.constraints:
            return np.zeros(3, dtype=np.float32)
        
        # 使用加权平均合并多个约束
        vectors = [c.to_vector() for c in self.constraints]
        weights = [c.weight for c in self.constraints]
        
        weighted_sum = np.sum([v * w for v, w in zip(vectors, weights)], axis=0)
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else vectors[0]

class ConstraintManager:
    """
    约束管理器
    
    提供约束的统一管理、验证和优化功能，支持参数化强化学习的约束注入
    """
    
    def __init__(self, 
                 freq_range: Tuple[float, float] = (0.5e9, 7e9),
                 s11_range: Tuple[float, float] = (-50.0, -3.0),
                 tolerance_range: Tuple[float, float] = (0.1, 10.0)):
        """
        初始化约束管理器
        
        Args:
            freq_range: 支持的频率范围 (Hz)
            s11_range: 支持的S11范围 (dB)
            tolerance_range: 支持的容差范围 (dB)
        """
        self.freq_range = freq_range
        self.s11_range = s11_range
        self.tolerance_range = tolerance_range
        
        # 约束存储
        self.constraints: Dict[str, ConstraintConfig] = {}
        self.constraint_groups: Dict[str, ConstraintGroup] = {}
        
        # 日志配置
        self.logger = logging.getLogger(__name__)
    
    def validate_constraint(self, constraint: ConstraintConfig) -> bool:
        """验证约束的物理合理性"""
        try:
            # 频率范围检查
            if not (self.freq_range[0] <= constraint.freq_low < constraint.freq_high <= self.freq_range[1]):
                self.logger.warning(f"频率范围超出支持范围: {constraint.freq_low/1e9:.2f}-{constraint.freq_high/1e9:.2f} GHz")
                return False
            
            # S11值检查
            if not (self.s11_range[0] <= constraint.target_s11 <= self.s11_range[1]):
                self.logger.warning(f"目标S11值超出范围: {constraint.target_s11} dB")
                return False
            
            # 容差检查
            if not (self.tolerance_range[0] <= constraint.tolerance <= self.tolerance_range[1]):
                self.logger.warning(f"容差超出范围: {constraint.tolerance} dB")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"约束验证失败: {e}")
            return False
    
    def add_constraint(self, constraint: ConstraintConfig, validate: bool = True) -> bool:
        """添加约束"""
        if validate and not self.validate_constraint(constraint):
            return False
        
        self.constraints[constraint.name] = constraint
        self.logger.info(f"添加约束: {constraint.name}")
        return True
    
    def get_constraint(self, name: str) -> Optional[ConstraintConfig]:
        """获取约束"""
        return self.constraints.get(name)
    
    def remove_constraint(self, name: str) -> bool:
        """移除约束"""
        if name in self.constraints:
            del self.constraints[name]
            self.logger.info(f"移除约束: {name}")
            return True
        return False
    
    def create_constraint_group(self, name: str, constraints: List[ConstraintConfig], 
                              priority: float = 1.0, description: str = "") -> bool:
        """创建约束组"""
        # 验证所有约束
        for constraint in constraints:
            if not self.validate_constraint(constraint):
                self.logger.error(f"约束组 {name} 包含无效约束")
                return False
        
        group = ConstraintGroup(
            name=name,
            constraints=constraints,
            priority=priority,
            description=description
        )
        
        self.constraint_groups[name] = group
        self.logger.info(f"创建约束组: {name}, 包含 {len(constraints)} 个约束")
        return True
    
    def get_constraint_group(self, name: str) -> Optional[ConstraintGroup]:
        """获取约束组"""
        return self.constraint_groups.get(name)
    
    def export_constraints(self, filepath: str, format: str = 'json'):
        """导出约束配置"""
        data = {
            'constraints': {name: asdict(constraint) for name, constraint in self.constraints.items()},
            'groups': {name: {
                'name': group.name,
                'constraints': [asdict(c) for c in group.constraints],
                'priority': group.priority,
                'description': group.description
            } for name, group in self.constraint_groups.items()}
        }
        
        if format.lower() == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"导出约束配置到: {filepath}")
    
    def import_constraints(self, filepath: str, format: str = 'json'):
        """导入约束配置"""
        if format.lower() == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 导入单个约束
            if 'constraints' in data:
                for name, constraint_data in data['constraints'].items():
                    constraint = ConstraintConfig(**constraint_data)
                    self.add_constraint(constraint, validate=False)
            
            # 导入约束组
            if 'groups' in data:
                for name, group_data in data['groups'].items():
                    constraints = [ConstraintConfig(**c) for c in group_data['constraints']]
                    self.create_constraint_group(
                        name=group_data['name'],
                        constraints=constraints,
                        priority=group_data.get('priority', 1.0),
                        description=group_data.get('description', '')
                    )
        
        self.logger.info(f"从文件导入约束配置: {filepath}")
    
    def get_all_constraints(self) -> List[ConstraintConfig]:
        """获取所有约束"""
        return list(self.constraints.values())
    
    def get_all_constraint_groups(self) -> List[ConstraintGroup]:
        """获取所有约束组"""
        return list(self.constraint_groups.values())
    
    def clear_all(self):
        """清空所有约束"""
        self.constraints.clear()
        self.constraint_groups.clear()
        self.logger.info("清空所有约束配置")