"""
约束管理器 - 统一管理约束配置、验证和转换
提供约束的生命周期管理和验证功能

核心功能:
1. 约束验证: 检查约束的物理合理性和一致性
2. 约束转换: 支持不同格式间的约束转换
3. 约束组合: 支持多约束组合和冲突检测
4. 约束优化: 自动优化约束参数以提高可达性
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from enum import Enum
import json
from pathlib import Path

# 导入相关模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constraint.types import ConstraintConfig

class ConstraintType(Enum):
    """约束类型枚举"""
    S11_SINGLE_BAND = "s11_single_band"      # 单频段S11约束
    S11_MULTI_BAND = "s11_multi_band"        # 多频段S11约束
    BANDWIDTH = "bandwidth"                   # 带宽约束
    RESONANCE = "resonance"                   # 谐振频率约束
    GAIN = "gain"                            # 增益约束
    EFFICIENCY = "efficiency"                 # 效率约束

class ValidationResult(Enum):
    """验证结果枚举"""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"

@dataclass
class ValidationReport:
    """验证报告"""
    result: ValidationResult
    messages: List[str]
    suggestions: List[str]
    
    def is_valid(self) -> bool:
        return self.result == ValidationResult.VALID
        
    def has_warnings(self) -> bool:
        return self.result == ValidationResult.WARNING

@dataclass
class ConstraintGroup:
    """约束组"""
    name: str
    constraints: List[ConstraintConfig]
    priority: float = 1.0
    description: str = ""
    
class ConstraintManager:
    """
    约束管理器
    
    提供约束的统一管理、验证和优化功能
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
        
        # 验证规则
        self.validation_rules = self._setup_validation_rules()
        
        # 日志配置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _setup_validation_rules(self) -> Dict[str, Any]:
        """设置验证规则"""
        return {
            'min_bandwidth': 0.1e9,      # 最小带宽100MHz
            'max_bandwidth': 4e9,        # 最大带宽4GHz
            'min_q_factor': 1.0,         # 最小Q因子
            'max_q_factor': 100.0,       # 最大Q因子
            'freq_resolution': 1e6,      # 频率分辨率1MHz
            's11_resolution': 0.1,       # S11分辨率0.1dB
            'physics_checks': True,      # 启用物理检查
            'consistency_checks': True   # 启用一致性检查
        }
        
    def add_constraint(self, constraint: ConstraintConfig, validate: bool = True) -> ValidationReport:
        """
        添加约束
        
        Args:
            constraint: 约束配置
            validate: 是否验证约束
            
        Returns:
            验证报告
        """
        if validate:
            report = self.validate_constraint(constraint)
            if not report.is_valid() and report.result == ValidationResult.INVALID:
                self.logger.error(f"约束验证失败: {constraint.name}")
                return report
                
        self.constraints[constraint.name] = constraint
        self.logger.info(f"添加约束: {constraint.name}")
        
        return ValidationReport(ValidationResult.VALID, [], [])
        
    def remove_constraint(self, name: str) -> bool:
        """移除约束"""
        if name in self.constraints:
            del self.constraints[name]
            self.logger.info(f"移除约束: {name}")
            return True
        return False
        
    def get_constraint(self, name: str) -> Optional[ConstraintConfig]:
        """获取约束"""
        return self.constraints.get(name)
        
    def list_constraints(self) -> List[str]:
        """列出所有约束名称"""
        return list(self.constraints.keys())
        
    def validate_constraint(self, constraint: ConstraintConfig) -> ValidationReport:
        """
        验证单个约束
        
        Args:
            constraint: 待验证的约束
            
        Returns:
            验证报告
        """
        messages = []
        suggestions = []
        result = ValidationResult.VALID
        
        # 1. 基本范围检查
        if not (self.freq_range[0] <= constraint.freq_low <= self.freq_range[1]):
            messages.append(f"频率下限超出范围: {constraint.freq_low/1e9:.1f}GHz")
            result = ValidationResult.INVALID
            
        if not (self.freq_range[0] <= constraint.freq_high <= self.freq_range[1]):
            messages.append(f"频率上限超出范围: {constraint.freq_high/1e9:.1f}GHz")
            result = ValidationResult.INVALID
            
        if not (self.s11_range[0] <= constraint.target_s11 <= self.s11_range[1]):
            messages.append(f"目标S11超出范围: {constraint.target_s11}dB")
            result = ValidationResult.INVALID
            
        if not (self.tolerance_range[0] <= constraint.tolerance <= self.tolerance_range[1]):
            messages.append(f"容差超出范围: {constraint.tolerance}dB")
            result = ValidationResult.INVALID
            
        # 2. 逻辑一致性检查
        if constraint.freq_high <= constraint.freq_low:
            messages.append("频率上限必须大于下限")
            result = ValidationResult.INVALID
            
        bandwidth = constraint.freq_high - constraint.freq_low
        if bandwidth < self.validation_rules['min_bandwidth']:
            messages.append(f"带宽过小: {bandwidth/1e9:.1f}GHz < {self.validation_rules['min_bandwidth']/1e9:.1f}GHz")
            result = ValidationResult.WARNING
            suggestions.append("考虑增加带宽以提高可实现性")
            
        if bandwidth > self.validation_rules['max_bandwidth']:
            messages.append(f"带宽过大: {bandwidth/1e9:.1f}GHz > {self.validation_rules['max_bandwidth']/1e9:.1f}GHz")
            result = ValidationResult.WARNING
            suggestions.append("考虑减小带宽以提高精度")
            
        # 3. 物理合理性检查
        if self.validation_rules['physics_checks']:
            physics_report = self._check_physics_constraints(constraint)
            messages.extend(physics_report['messages'])
            suggestions.extend(physics_report['suggestions'])
            if physics_report['severity'] == 'error':
                result = ValidationResult.INVALID
            elif physics_report['severity'] == 'warning' and result == ValidationResult.VALID:
                result = ValidationResult.WARNING
                
        # 4. Q因子检查
        center_freq = (constraint.freq_low + constraint.freq_high) / 2
        q_factor = center_freq / bandwidth
        
        if q_factor < self.validation_rules['min_q_factor']:
            messages.append(f"Q因子过低: {q_factor:.1f}")
            result = ValidationResult.WARNING
            suggestions.append("考虑减小带宽以提高Q因子")
            
        if q_factor > self.validation_rules['max_q_factor']:
            messages.append(f"Q因子过高: {q_factor:.1f}")
            result = ValidationResult.WARNING
            suggestions.append("考虑增加带宽以降低Q因子")
            
        return ValidationReport(result, messages, suggestions)
        
    def _check_physics_constraints(self, constraint: ConstraintConfig) -> Dict[str, Any]:
        """检查物理约束"""
        messages = []
        suggestions = []
        severity = 'info'
        
        center_freq = (constraint.freq_low + constraint.freq_high) / 2
        bandwidth = constraint.freq_high - constraint.freq_low
        
        # 1. 高频段S11目标检查
        if center_freq > 3e9 and constraint.target_s11 > -8.0:
            messages.append("高频段(>3GHz)建议S11目标更严格(<-8dB)")
            suggestions.append("考虑将目标S11设置为-10dB或更低")
            severity = 'warning'
            
        # 2. 窄带约束检查
        if bandwidth < 0.3e9 and constraint.tolerance > 3.0:
            messages.append("窄带约束建议使用更小的容差")
            suggestions.append("考虑将容差设置为2dB以下")
            severity = 'warning'
            
        # 3. 宽带约束检查
        if bandwidth > 2e9 and constraint.target_s11 < -20.0:
            messages.append("宽带约束的严格S11目标可能难以实现")
            suggestions.append("考虑放宽S11目标到-15dB左右")
            severity = 'warning'
            
        # 4. 频段特性检查
        if constraint.freq_low < 1e9:
            messages.append("低频段(<1GHz)可能需要更大的天线尺寸")
            suggestions.append("确保天线尺寸足够支持低频工作")
            severity = 'warning'
            
        if constraint.freq_high > 5e9:
            messages.append("高频段(>5GHz)可能受制造精度影响")
            suggestions.append("考虑制造容差对高频性能的影响")
            severity = 'warning'
            
        return {
            'messages': messages,
            'suggestions': suggestions,
            'severity': severity
        }
        
    def validate_constraint_group(self, group: ConstraintGroup) -> ValidationReport:
        """验证约束组"""
        all_messages = []
        all_suggestions = []
        worst_result = ValidationResult.VALID
        
        # 验证每个约束
        for constraint in group.constraints:
            report = self.validate_constraint(constraint)
            all_messages.extend([f"[{constraint.name}] {msg}" for msg in report.messages])
            all_suggestions.extend([f"[{constraint.name}] {sug}" for sug in report.suggestions])
            
            if report.result == ValidationResult.INVALID:
                worst_result = ValidationResult.INVALID
            elif report.result == ValidationResult.WARNING and worst_result == ValidationResult.VALID:
                worst_result = ValidationResult.WARNING
                
        # 检查约束间冲突
        conflict_report = self._check_constraint_conflicts(group.constraints)
        all_messages.extend(conflict_report['messages'])
        all_suggestions.extend(conflict_report['suggestions'])
        
        if conflict_report['has_conflicts']:
            worst_result = ValidationResult.INVALID
            
        return ValidationReport(worst_result, all_messages, all_suggestions)
        
    def _check_constraint_conflicts(self, constraints: List[ConstraintConfig]) -> Dict[str, Any]:
        """检查约束间冲突"""
        messages = []
        suggestions = []
        has_conflicts = False
        
        if len(constraints) < 2:
            return {'messages': messages, 'suggestions': suggestions, 'has_conflicts': has_conflicts}
            
        # 检查频段重叠
        for i, c1 in enumerate(constraints):
            for j, c2 in enumerate(constraints[i+1:], i+1):
                # 频段重叠检查
                overlap_low = max(c1.freq_low, c2.freq_low)
                overlap_high = min(c1.freq_high, c2.freq_high)
                
                if overlap_low < overlap_high:
                    overlap_bw = overlap_high - overlap_low
                    messages.append(f"约束{c1.name}和{c2.name}频段重叠: {overlap_bw/1e9:.1f}GHz")
                    
                    # 检查重叠区域的S11目标是否兼容
                    s11_diff = abs(c1.target_s11 - c2.target_s11)
                    combined_tolerance = c1.tolerance + c2.tolerance
                    
                    if s11_diff > combined_tolerance:
                        messages.append(f"重叠区域S11目标冲突: {s11_diff:.1f}dB > {combined_tolerance:.1f}dB")
                        suggestions.append(f"调整{c1.name}或{c2.name}的S11目标或容差")
                        has_conflicts = True
                    else:
                        suggestions.append("重叠区域目标兼容，但可能影响优化难度")
                        
        return {'messages': messages, 'suggestions': suggestions, 'has_conflicts': has_conflicts}
        
    def create_constraint_group(self, 
                              name: str, 
                              constraints: List[ConstraintConfig],
                              priority: float = 1.0,
                              description: str = "") -> ValidationReport:
        """创建约束组"""
        group = ConstraintGroup(name, constraints, priority, description)
        report = self.validate_constraint_group(group)
        
        if report.is_valid() or report.has_warnings():
            self.constraint_groups[name] = group
            self.logger.info(f"创建约束组: {name}, 包含{len(constraints)}个约束")
            
        return report
        
    def optimize_constraint(self, constraint: ConstraintConfig) -> ConstraintConfig:
        """优化约束参数以提高可达性"""
        optimized = ConstraintConfig(
            freq_low=constraint.freq_low,
            freq_high=constraint.freq_high,
            target_s11=constraint.target_s11,
            tolerance=constraint.tolerance,
            weight=constraint.weight,
            name=f"{constraint.name}_optimized"
        )
        
        # 1. 基于频段调整S11目标
        center_freq = (constraint.freq_low + constraint.freq_high) / 2
        if center_freq > 3e9:
            # 高频段：放宽S11目标
            optimized.target_s11 = max(optimized.target_s11, -12.0)
        elif center_freq < 1e9:
            # 低频段：可以更严格
            optimized.target_s11 = min(optimized.target_s11, -15.0)
            
        # 2. 基于带宽调整容差
        bandwidth = constraint.freq_high - constraint.freq_low
        if bandwidth < 0.5e9:
            # 窄带：减小容差
            optimized.tolerance = min(optimized.tolerance, 2.0)
        elif bandwidth > 2e9:
            # 宽带：增加容差
            optimized.tolerance = max(optimized.tolerance, 3.0)
            
        # 3. Q因子优化
        q_factor = center_freq / bandwidth
        if q_factor > 50:
            # Q因子过高，增加带宽
            new_bandwidth = center_freq / 30  # 目标Q=30
            freq_center = (optimized.freq_low + optimized.freq_high) / 2
            optimized.freq_low = freq_center - new_bandwidth / 2
            optimized.freq_high = freq_center + new_bandwidth / 2
            
        self.logger.info(f"约束优化完成: {constraint.name} -> {optimized.name}")
        return optimized
        
    def convert_to_dict(self, constraint: ConstraintConfig) -> Dict[str, Any]:
        """将约束转换为字典格式"""
        return asdict(constraint)
        
    def convert_from_dict(self, data: Dict[str, Any]) -> ConstraintConfig:
        """从字典创建约束"""
        return ConstraintConfig(**data)
        
    def export_constraints(self, filepath: str, format: str = 'json'):
        """导出约束配置"""
        if format.lower() == 'json':
            data = {
                'constraints': {name: asdict(constraint) for name, constraint in self.constraints.items()},
                'groups': {name: {
                    'name': group.name,
                    'constraints': [asdict(c) for c in group.constraints],
                    'priority': group.priority,
                    'description': group.description
                } for name, group in self.constraint_groups.items()}
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        self.logger.info(f"约束配置已导出到: {filepath}")
        
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
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取约束统计信息"""
        if not self.constraints:
            return {}
            
        constraints = list(self.constraints.values())
        
        freq_lows = [c.freq_low for c in constraints]
        freq_highs = [c.freq_high for c in constraints]
        bandwidths = [c.freq_high - c.freq_low for c in constraints]
        targets = [c.target_s11 for c in constraints]
        tolerances = [c.tolerance for c in constraints]
        
        return {
            'total_constraints': len(constraints),
            'total_groups': len(self.constraint_groups),
            'frequency_coverage': {
                'min_freq': min(freq_lows) / 1e9,
                'max_freq': max(freq_highs) / 1e9,
                'total_span': (max(freq_highs) - min(freq_lows)) / 1e9
            },
            'bandwidth_stats': {
                'min_bandwidth': min(bandwidths) / 1e9,
                'max_bandwidth': max(bandwidths) / 1e9,
                'avg_bandwidth': np.mean(bandwidths) / 1e9
            },
            's11_stats': {
                'min_target': min(targets),
                'max_target': max(targets),
                'avg_target': np.mean(targets)
            },
            'tolerance_stats': {
                'min_tolerance': min(tolerances),
                'max_tolerance': max(tolerances),
                'avg_tolerance': np.mean(tolerances)
            }
        }