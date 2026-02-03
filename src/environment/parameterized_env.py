"""
参数化像素天线强化学习环境

实现支持动态约束注入的泛化强化学习环境，核心特性：
1. 参数化状态空间：[像素配置, S11观测, 物理特征, 约束向量]
2. 动态约束注入：支持任意频段和目标S11值的实时配置
3. 泛化奖励函数：基于约束参数的自适应奖励计算
4. 物理特征提取：谐振频率、带宽等电磁特征增强状态表示
5. 缓存优化：智能缓存机制提升仿真效率
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import re
import json
import hashlib
from datetime import datetime
import warnings

# 导入核心模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api import HFSSController
from src.config.constraint_config import ConstraintConfig, ConstraintGroup

class ParameterizedPixelAntennaEnv(gym.Env):
    """
    通用参数化设计强化学习环境
    
    观测空间: [参数向量(99维) + S11采样(20点) + 物理特征(4维) + 约束向量(3维)]
    动作空间: 支持连续向量或二值参数向量
    奖励函数: 基于约束参数的动态奖励计算
    """
    
    def __init__(self, 
                 project_path: str,
                 design_name: str = "HFSSDesign1",
                 setup_name: str = "Setup1",
                 sweep_name: str = "Sweep",
                 grid_size: Tuple[int, int] = (10, 10),
                 freq_samples: int = 20,
                 max_steps: int = 50,
                 default_constraint: Optional[ConstraintConfig] = None,
                 action_mode: str = 'discrete',
                 variable_bounds: Tuple[float, float] = (0.001, 2.0),
                 offline: bool = False,
                 sim_freq_range: Tuple[float, float] = (2e9, 7e9),
                 out_of_band_threshold: float = -5.0,
                 max_bands: int = 3,
                 param_count: int = 99,
                 allowed_param_values: Optional[List[float]] = None):
        """
        初始化参数化环境
        
        Args:
            project_path: HFSS项目路径
            design_name: 设计名称
            setup_name: 仿真设置名称
            sweep_name: 扫频名称
            grid_size: 像素网格大小
            freq_samples: S11频率采样点数
            max_steps: 最大步数
            default_constraint: 默认约束配置
            action_mode: 动作模式，'discrete'或'continuous'
            variable_bounds: 参数向量边界 (min, max)
            offline: 是否为离线模式
            sim_freq_range: 全局仿真频率范围 (Hz)
            out_of_band_threshold: 带外S11阈值 (dB)，带外要求为S11应高于该阈值
        """
        super().__init__()
        
        # 环境配置
        self.project_path = project_path
        self.design_name = design_name
        self.setup_name = setup_name
        self.sweep_name = sweep_name
        self.grid_size = grid_size
        self.freq_samples = freq_samples
        self.max_steps = max_steps
        self.action_mode = action_mode
        self.variable_bounds = variable_bounds
        
        # 运行模式与约束相关配置
        self.offline = offline
        self.sim_freq_range = sim_freq_range
        self.out_of_band_threshold = out_of_band_threshold
        self.max_bands = int(max_bands)
        self.constraint_vector_dim = self.max_bands * 3 + 1
        
        # 参数向量配置：支持自定义维度
        self.param_count = int(param_count)
        self.pixel_count = self.param_count  # 兼容旧字段
        self.current_params = np.ones(self.param_count, dtype=np.float32) * self.variable_bounds[0]
        
        # 允许的离散取值列表（用于list/multidiscrete模式）
        self.allowed_param_values = None
        if allowed_param_values is not None:
            try:
                vals = [float(v) for v in list(allowed_param_values)]
                if len(vals) >= 2:
                    self.allowed_param_values = vals
            except Exception:
                self.allowed_param_values = None
        
        # 约束配置
        self.current_constraint = default_constraint or ConstraintConfig(
            freq_low=1e9, freq_high=3e9, target_s11=-10.0
        )
        self.current_constraint_group: Optional[ConstraintGroup] = None
        
        # 状态变量
        self.current_step = 0
        self.s11_data = None
        self.frequencies = None
        self.physical_features = np.zeros(4)  # [谐振频率, 带宽, 最小S11, 偏差]
        self.global_step = 0
        self._used_cache = False
        self._last_param_sig = None
        self.last_csv_path = None
        # 在连接HFSS之前先初始化控制器和日志器
        self.hfss_controller = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 连接HFSS（如果尚未连接）
        if (not self.offline) and (self.hfss_controller is None):
            self._connect_hfss()
            
        # 日志与目录配置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 输出与缓存目录
        try:
            base_outputs = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / 'outputs'
            (base_outputs / 's_params_cache').mkdir(parents=True, exist_ok=True)
            self.output_dir = base_outputs
            self.cache_dir = base_outputs / 's_params_cache'
        except Exception:
            self.output_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.cache_dir = self.output_dir
        
        # 日志文件路径
        self.param_log_path = self.output_dir / 'param_history.csv'
        self.reward_log_path = self.output_dir / 'training_rewards.csv'
        self.reward_log_friendly_path = self.output_dir / 'training_rewards_readable.csv'
        self.sparams_index_path = self.output_dir / 's_params_index.csv'
        
        # 初始化日志文件头
        try:
            if not self.param_log_path.exists():
                with open(self.param_log_path, 'w', encoding='utf-8') as f:
                    f.write('timestamp,step,action_mode,param_hash,params_json,constraint_name,constraint_range,target_s11,tolerance\n')
            if not self.reward_log_path.exists():
                with open(self.reward_log_path, 'w', encoding='utf-8') as f:
                    f.write('timestamp,step,param_hash,reward,constraint_name,freq_low,freq_high,target_s11,tolerance\n')
            if not self.reward_log_friendly_path.exists():
                with open(self.reward_log_friendly_path, 'w', encoding='utf-8') as f:
                    f.write('timestamp,global_step,episode_step,param_hash,reward,constraint_name,freq_low_GHz,freq_high_GHz,target_s11,tolerance\n')
            if not self.sparams_index_path.exists():
                with open(self.sparams_index_path, 'w', encoding='utf-8') as f:
                    f.write('timestamp,global_step,param_hash,constraint_name,s_params_cache,s_params_cache_simple\n')
        except Exception:
            pass
        
        # 缓存配置
        self.enable_cache = True
        self.cache_skip_hfss_update_if_hit = True
        
        # 控制器与空间
        self.hfss_controller = None
        self._setup_spaces()
        
    def set_constraint(self, constraint: ConstraintConfig):
        """设置新的约束配置"""
        # 兼容单约束或约束组
        if isinstance(constraint, ConstraintGroup):
            self.current_constraint_group = constraint
            # 用组内首个约束作为当前日志参考
            first = constraint.constraints[0] if constraint.constraints else self.current_constraint
            self.current_constraint = first
            self.logger.info(
                f"设置约束组: {constraint.name}, 频段数={len(constraint.constraints)}"
            )
        else:
            self.current_constraint_group = None
            self.current_constraint = constraint
            self.logger.info(
                f"设置新约束: {constraint.freq_low/1e9:.1f}-{constraint.freq_high/1e9:.1f}GHz, 目标S11={constraint.target_s11}dB"
            )
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置状态
        self.current_step = 0
        self.current_params = np.ones(self.param_count, dtype=np.float32) * self.variable_bounds[0]
        self.s11_data = np.ones(self.freq_samples) * (-5.0)  # 初始S11假设为-5dB
        self.physical_features = np.zeros(4)
        self._used_cache = False
        self._last_param_sig = None
        self.last_csv_path = None
        
        # 如果提供了新约束，更新约束配置
        if options and 'constraint' in options:
            self.set_constraint(options['constraint'])
            
        # 连接HFSS（如果尚未连接）
        if (not self.offline) and (self.hfss_controller is None):
            self._connect_hfss()
            
        # 获取初始观测
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
        
    def step(self, action):
        """执行动作"""
        mode = getattr(self, 'action_mode', 'discrete')
        # 连续模式：直接设置参数向量k
        if mode == 'continuous':
            action = np.asarray(action, dtype=np.float32)
            low, high = self.variable_bounds
            if action.shape != (self.param_count,):
                action = np.full(self.param_count, float(action), dtype=np.float32)
            self.current_params = np.clip(action, low, high)
        elif mode == 'binary':
            action = np.asarray(action, dtype=np.int8)
            if action.shape != (self.param_count,):
                action = np.zeros(self.param_count, dtype=np.int8)
            low, high = self.variable_bounds
            self.current_params = np.where(action > 0, high, low).astype(np.float32)
        elif mode in ('list', 'multidiscrete') and (self.allowed_param_values is not None):
            # MultiDiscrete索引映射到允许的参数值列表
            idx = np.asarray(action, dtype=np.int64)
            if idx.shape != (self.param_count,):
                idx = np.zeros(self.param_count, dtype=np.int64)
            ncat = len(self.allowed_param_values)
            idx = np.clip(idx, 0, ncat - 1)
            vals = np.take(np.asarray(self.allowed_param_values, dtype=np.float32), idx)
            self.current_params = vals
        else:
            # 旧的离散翻转（单点翻转）
            if isinstance(action, np.ndarray):
                action = action.item()  # 从numpy数组中提取标量值
            action = int(action)  # 确保action是整数类型
            if self.current_params[action] == self.variable_bounds[0]:
                self.current_params[action] = self.variable_bounds[1]
            else:
                self.current_params[action] = self.variable_bounds[0]
        
        # 记录参数签名与日志
        self._last_param_sig = self._compute_param_signature()
        self._log_current_params(self._last_param_sig)
        
        self.current_step += 1
        self.global_step += 1
        
        # 更新HFSS模型并运行仿真（或使用缓存）
        success = self._update_hfss_model()
        
        if success:
            # 获取S11数据
            self._get_s11_data()
            # 提取物理特征
            self._extract_physical_features()
        else:
            # 仿真失败，给予惩罚
            self.logger.warning("HFSS仿真失败")
        
        # 计算奖励
        reward = self._calculate_reward()
        # 奖励日志
        try:
            self._log_step_reward(reward)
        except Exception:
            pass
        
        # 检查终止条件
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        # 获取观测和信息
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
        
    def _connect_hfss(self):
        """连接HFSS"""
        try:
            self.hfss_controller = HFSSController(
                project_path=self.project_path,
                design_name=self.design_name,
                setup_name=self.setup_name,
                sweep_name=self.sweep_name
            )
            self.hfss_controller.connect()
            self.logger.info("HFSS连接成功")
        except Exception as e:
            self.logger.error(f"HFSS连接失败: {e}")
            self.hfss_controller = None
            
    def _update_hfss_model(self) -> bool:
        """更新HFSS模型中的参数向量"""
        # 命中缓存则跳过HFSS更新与仿真
        if self.enable_cache and self._last_param_sig:
            cache_path = self.cache_dir / f"s11_{self._last_param_sig}.csv"
            if cache_path.exists():
                self.last_csv_path = cache_path
                self._used_cache = True
                self.logger.debug("命中仿真缓存，跳过HFSS仿真")
                if self.cache_skip_hfss_update_if_hit:
                    return True
        
        if self.offline:
            return True
        if self.hfss_controller is None:
            return False
            
        try:
            # 直接使用set_variable设置k参数列表（单位mm）
            success = self.hfss_controller.set_variable("k", self.current_params, unit="mm")
            if not success:
                self.logger.error("参数设置失败")
                return False
                
            # 运行仿真
            result = self.hfss_controller.analyze()
            return result
            
        except Exception as e:
            self.logger.error(f"HFSS模型更新失败: {e}")
            return False
            
    def _get_s11_data(self):
        """获取S11数据（CSV优先）"""
        # 若命中缓存，直接读取缓存CSV
        if self._used_cache and self.last_csv_path and self.last_csv_path.exists():
            try:
                data = pd.read_csv(self.last_csv_path)
                freq_col = None
                for c in data.columns:
                    lc = c.lower()
                    if lc.startswith("freq"):
                        freq_col = c
                        break
                if freq_col is None:
                    raise ValueError("CSV缺少频率列")
                freq_vals = data[freq_col].values.astype(float)
                if "ghz" in freq_col.lower():
                    freq_vals = freq_vals * 1e9
                elif np.nanmax(freq_vals) < 1e6:
                    freq_vals = freq_vals * 1e9
                db_cols = [c for c in data.columns if "db(" in c.lower()]
                if len(db_cols) == 0:
                    raise ValueError("CSV缺少S11列")
                s11_vals = data[db_cols[0]].values.astype(float)
                self.frequencies = np.linspace(np.nanmin(freq_vals), np.nanmax(freq_vals), self.freq_samples)
                ok = ~np.isnan(s11_vals) & ~np.isnan(freq_vals)
                self.s11_data = np.interp(self.frequencies, freq_vals[ok], s11_vals[ok])
                return
            except Exception as e:
                self.logger.warning(f"缓存读取失败: {e}; 回退到常规数据获取流程")
        
        # Offline仿真：生成可重复的模拟曲线，便于快速测试
        if self.offline:
            rng = np.random.default_rng(seed=42 + self.current_step)
            if self.frequencies is None:
                fr_low, fr_high = self.sim_freq_range
                self.frequencies = np.linspace(fr_low, fr_high, self.freq_samples)
            center = (self.current_constraint.freq_low + self.current_constraint.freq_high) / 2.0
            bw = (self.current_constraint.freq_high - self.current_constraint.freq_low) / 6.0
            s11 = -5.0 - 15.0 * np.exp(-((self.frequencies - center) ** 2) / (2 * bw * bw))
            noise = rng.normal(0.0, 0.5, size=self.freq_samples)
            self.s11_data = s11 + noise
            
            try:
                cache_path = self.cache_dir / f"s11_{self._last_param_sig}.csv"
                df_cache = pd.DataFrame({
                    'Frequency(Hz)': self.frequencies,
                    'dB(S(1:1,1:1))': self.s11_data
                })
                df_cache.to_csv(cache_path, index=False)
                simple_cache_path = self.cache_dir / f"s11_{self._last_param_sig}_simple.csv"
                pd.DataFrame({
                    'Freq_GHz': np.round(self.frequencies / 1e9, 3),
                    'S11_dB': np.round(self.s11_data, 2)
                }).to_csv(simple_cache_path, index=False)
                self.last_csv_path = cache_path
                self._log_s_params_index(self._last_param_sig, str(cache_path), str(simple_cache_path))
            except Exception:
                pass
            return

        if self.hfss_controller is None:
            return

        try:
            # 获取S参数数据（dB），然后保存CSV并从CSV读取以确保一致性
            df = self.hfss_controller.get_s_params(
                port_combinations=[('1', '1')],
                data_format="dB",
            )

            if df is not None and not df.empty:
                # 直接从内存df解析频率与S11（保留原分辨率）
                try:
                    data = df
                    freq_col = None
                    for c in data.columns:
                        lc = c.lower()
                        if lc.startswith('freq'):
                            freq_col = c
                            break
                    if freq_col is None:
                        freq_col = 'Frequency'
                    freq_vals = data[freq_col].values.astype(float)
                    if 'ghz' in freq_col.lower() or np.nanmax(freq_vals) < 1e6:
                        freq_vals = freq_vals * 1e9
                    db_cols = [c for c in data.columns if 'db(' in c.lower()]
                    if len(db_cols) == 0:
                        s_cols = [c for c in data.columns if c.lower().startswith('s(')]
                        if len(s_cols) == 0:
                            raise ValueError('CSV缺少S11列')
                        raw = data[s_cols[0]].astype(str).values
                        def to_db(v: str) -> float:
                            try:
                                s = v.strip().strip('()')
                                if ',' in s:
                                    a, b = s.split(',', 1)
                                    comp = complex(float(a), float(b))
                                else:
                                    comp = complex(s.replace(' ', ''))
                                mag = np.abs(comp)
                                return 20 * np.log10(mag + 1e-12)
                            except Exception:
                                return np.nan
                        s11_vals = np.array([to_db(v) for v in raw], dtype=float)
                    else:
                        s11_vals = data[db_cols[0]].values.astype(float)
                    # 观测用插值到固定采样点
                    self.frequencies = np.linspace(np.nanmin(freq_vals), np.nanmax(freq_vals), self.freq_samples)
                    ok = ~np.isnan(s11_vals) & ~np.isnan(freq_vals)
                    self.s11_data = np.interp(self.frequencies, freq_vals[ok], s11_vals[ok])
                    # 写入缓存（保留原分辨率）
                    cache_path = self.cache_dir / f's11_{self._last_param_sig}.csv'
                    df_cache = pd.DataFrame({
                        'Frequency(Hz)': freq_vals,
                        'dB(S(1:1,1:1))': s11_vals
                    })
                    df_cache.to_csv(cache_path, index=False)
                    # 简化副本（GHz与固定小数）
                    simple_cache_path = self.cache_dir / f's11_{self._last_param_sig}_simple.csv'
                    df_simple = pd.DataFrame({
                        'Freq_GHz': np.round(freq_vals / 1e9, 3),
                        'S11_dB': np.round(s11_vals, 2)
                    })
                    df_simple.to_csv(simple_cache_path, index=False)
                    self.last_csv_path = cache_path
                    # 索引映射
                    self._log_s_params_index(self._last_param_sig, str(cache_path), str(simple_cache_path))
                except Exception as e:
                    self.logger.warning(f'S11数据解析失败: {e}')
            else:
                self.logger.warning("S11数据获取失败，使用默认值")
                self.s11_data = np.ones(self.freq_samples) * (-5.0)
                if self.frequencies is None:
                    self.frequencies = np.linspace(self.current_constraint.freq_low, self.current_constraint.freq_high, self.freq_samples)

        except Exception as e:
            self.logger.error(f"S11数据获取错误: {e}")
            self.s11_data = np.ones(self.freq_samples) * (-5.0)
            if self.frequencies is None:
                self.frequencies = np.linspace(self.current_constraint.freq_low, self.current_constraint.freq_high, self.freq_samples)
            
    def _extract_physical_features(self):
        """提取物理特征"""
        if self.s11_data is None or self.frequencies is None:
            return
            
        try:
            # 1. 谐振频率（S11最小值对应的频率）
            min_idx = np.argmin(self.s11_data)
            resonant_freq = self.frequencies[min_idx]
            
            # 2. 带宽（-10dB带宽）
            threshold = -10.0
            below_threshold = self.s11_data < threshold
            if np.any(below_threshold):
                indices = np.where(below_threshold)[0]
                bandwidth = self.frequencies[indices[-1]] - self.frequencies[indices[0]]
            else:
                bandwidth = 0.0
                
            # 3. 最小S11值
            min_s11 = np.min(self.s11_data)
            
            # 4. 约束偏差（在约束频段内的平均偏差）
            constraint_mask = (self.frequencies >= self.current_constraint.freq_low) & \
                            (self.frequencies <= self.current_constraint.freq_high)
            if np.any(constraint_mask):
                constraint_s11 = self.s11_data[constraint_mask]
                deviation = np.mean(np.abs(constraint_s11 - self.current_constraint.target_s11))
            else:
                deviation = 100.0  # 大偏差
                
            # 归一化特征
            self.physical_features = np.array([
                resonant_freq / 1e9,  # GHz
                bandwidth / 1e9,      # GHz
                min_s11 / (-50.0),    # 归一化到[-50dB, 0dB]
                deviation / 50.0      # 归一化偏差
            ], dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"物理特征提取错误: {e}")
            self.physical_features = np.zeros(4, dtype=np.float32)
            
    def _calculate_reward(self) -> float:
        """计算基于约束的动态奖励"""
        if self.s11_data is None or self.frequencies is None:
            return -10.0
        
        # 构造频段掩码（支持约束组）
        bands = []
        if self.current_constraint_group is not None:
            bands = self.current_constraint_group.constraints
        else:
            bands = [self.current_constraint]
        
        # 联合带外掩码
        mask_union = np.zeros_like(self.frequencies, dtype=bool)
        inband_reward = 0.0
        performance_reward = 0.0
        total_weight = 0.0
        
        for c in bands:
            band_mask = (self.frequencies >= c.freq_low) & (self.frequencies <= c.freq_high)
            if not np.any(band_mask):
                # 无有效频段，给予小惩罚但继续
                performance_reward -= 1.0
                continue
            mask_union |= band_mask
            s11_band = self.s11_data[band_mask]
            target = c.target_s11
            tol = c.tolerance
            w = getattr(c, 'weight', 1.0)
            total_weight += w
            
            # 满足度: 不等式满足比例 (S11 <= target + tol)
            satisfied = np.sum(s11_band <= (target + tol))
            satisfaction_ratio = satisfied / len(s11_band)
            inband_reward += w * (satisfaction_ratio * 10.0)
            
            # 违背程度: 超过目标的平均超量 (S11 - target 的正部)
            excess = np.maximum(0.0, s11_band - target)
            mean_excess = float(np.mean(excess))
            performance_reward += w * max(0.0, 10.0 - mean_excess)
        
        if total_weight > 0:
            inband_reward /= total_weight
            performance_reward /= total_weight
        
        # 带外惩罚：非任何频段的点，如果S11低于阈值则惩罚
        out_mask = ~mask_union
        out_penalty = 0.0
        if np.any(out_mask):
            out_s11 = self.s11_data[out_mask]
            bad_points = np.sum(out_s11 < self.out_of_band_threshold)
            out_penalty = bad_points * (-0.5)
        
        # 物理合理性奖励
        physics_reward = 0.0
        if self.physical_features[1] > 0:  # 有带宽
            physics_reward += 1.0
        if self.physical_features[2] > 0.2:  # 有一定的S11深度
            physics_reward += 1.0
            
        # 总奖励
        total_reward = inband_reward + performance_reward + out_penalty + physics_reward
        
        return float(total_reward)
        
    def _is_terminated(self) -> bool:
        """检查是否达到终止条件"""
        if self.s11_data is None or self.frequencies is None:
            return False
        
        bands = []
        if self.current_constraint_group is not None:
            bands = self.current_constraint_group.constraints
        else:
            bands = [self.current_constraint]
        
        # 所有频段均满足则终止
        for c in bands:
            band_mask = (self.frequencies >= c.freq_low) & (self.frequencies <= c.freq_high)
            if not np.any(band_mask):
                return False
            s11_band = self.s11_data[band_mask]
            if not np.all(s11_band <= (c.target_s11 + c.tolerance)):
                return False
        return True
        
    def _get_observation(self) -> np.ndarray:
        """获取当前观测"""
        # 参数向量
        param_obs = self.current_params.astype(np.float32)
        
        # S11数据
        s11_obs = self.s11_data if self.s11_data is not None else np.ones(self.freq_samples) * (-5.0)
        s11_obs = s11_obs.astype(np.float32)
        
        # 物理特征
        physics_obs = self.physical_features.astype(np.float32)
        
        # 约束向量：按max_bands填充 [fl, fh, target] * max_bands + [oob_threshold]
        constraint_vec = []
        if self.current_constraint_group is not None:
            bands = self.current_constraint_group.constraints
        else:
            bands = [self.current_constraint]
        
        for i in range(self.max_bands):
            if i < len(bands):
                c = bands[i]
                constraint_vec.extend([
                    c.freq_low / 1e9,
                    c.freq_high / 1e9,
                    c.target_s11 / (-50.0)
                ])
            else:
                constraint_vec.extend([0.0, 0.0, 0.0])
        # 带外阈值（归一化到[-50,0]）
        constraint_vec.append(self.out_of_band_threshold / (-50.0))
        constraint_obs = np.array(constraint_vec, dtype=np.float32)
        
        observation = np.concatenate([param_obs, s11_obs, physics_obs, constraint_obs])
        return observation
        
    def _get_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        info = {
            'step': self.current_step,
            'constraint': {
                'freq_low': self.current_constraint.freq_low,
                'freq_high': self.current_constraint.freq_high,
                'target_s11': self.current_constraint.target_s11
            },
            'physical_features': {
                'resonant_freq': self.physical_features[0] * 1e9 if len(self.physical_features) > 0 else 0,
                'bandwidth': self.physical_features[1] * 1e9 if len(self.physical_features) > 1 else 0,
                'min_s11': self.physical_features[2] * (-50.0) if len(self.physical_features) > 2 else 0,
                'deviation': self.physical_features[3] * 50.0 if len(self.physical_features) > 3 else 0
            },
            'param_sum_mm': float(np.sum(self.current_params)),
            's11_available': self.s11_data is not None
        }
        if self.current_constraint_group is not None:
            info['constraint_group'] = {
                'name': self.current_constraint_group.name,
                'n_bands': len(self.current_constraint_group.constraints)
            }
        return info
        
    def render(self, mode='human'):
        """渲染环境（可选实现）"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Params: {np.sum(self.current_params)}/{self.param_count}")
            if self.current_constraint_group is not None:
                print(f"ConstraintGroup: {self.current_constraint_group.name}, Bands={len(self.current_constraint_group.constraints)}, OOB>{self.out_of_band_threshold}dB")
            else:
                print(f"Constraint: {self.current_constraint.freq_low/1e9:.1f}-{self.current_constraint.freq_high/1e9:.1f}GHz, Target: {self.current_constraint.target_s11}dB")
            if self.s11_data is not None:
                print(f"S11 range: [{np.min(self.s11_data):.1f}, {np.max(self.s11_data):.1f}]dB")
    def _compute_param_signature(self) -> str:
        vals = [float(v) for v in np.asarray(self.current_params).tolist()]
        # 固定量化，避免微小浮点噪声导致签名不同
        quantized = [round(v, 6) for v in vals]
        payload = {
            'mode': getattr(self, 'action_mode', 'discrete'),
            'bounds': list(self.variable_bounds),
            'params': quantized
        }
        s = json.dumps(payload, separators=(',', ':'), ensure_ascii=False)
        return hashlib.sha1(s.encode('utf-8')).hexdigest()

    def _log_current_params(self, sig: str):
        try:
            constraint_name = getattr(self.current_constraint, 'name', 'N/A')
            fr = f"{self.current_constraint.freq_low:.3e}-{self.current_constraint.freq_high:.3e}"
            ts = datetime.now().isoformat(timespec='seconds')
            row = [ts, str(self.current_step), getattr(self, 'action_mode', 'discrete'), sig,
                   json.dumps([float(v) for v in self.current_params.tolist()]),
                   constraint_name, fr, str(self.current_constraint.target_s11), str(self.current_constraint.tolerance)]
            with open(self.param_log_path, 'a', encoding='utf-8') as f:
                f.write(','.join(row) + '\n')
        except Exception:
            pass

    def _log_step_reward(self, reward: float):
        ts = datetime.now().isoformat(timespec='seconds')
        sig = self._last_param_sig or ''
        c = self.current_constraint
        row = [
            ts,
            str(self.current_step),
            sig,
            f"{reward:.6f}",
            getattr(c, 'name', 'N/A'),
            f"{getattr(c,'freq_low',0.0):.6e}",
            f"{getattr(c,'freq_high',0.0):.6e}",
            f"{getattr(c,'target_s11',0.0)}",
            f"{getattr(c,'tolerance',0.0)}",
        ]
        with open(self.reward_log_path, 'a', encoding='utf-8') as f:
            f.write(','.join(row) + '\n')
        # 友好版奖励日志（包含全局步数和固定小数格式）
        try:
            self.reward_log_friendly_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / 'outputs' / 'training_rewards_readable.csv'
            if not self.reward_log_friendly_path.exists():
                with open(self.reward_log_friendly_path, 'w', encoding='utf-8') as f:
                    f.write('timestamp,global_step,episode_step,param_hash,reward,constraint_name,freq_low_GHz,freq_high_GHz,target_s11,tolerance\n')
        except Exception:
            pass
        # 奖励日志
        try:
            self.reward_log_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / 'outputs' / 'training_rewards.csv'
            if not self.reward_log_path.exists():
                with open(self.reward_log_path, 'w', encoding='utf-8') as f:
                    f.write('timestamp,step,param_hash,reward,constraint_name,freq_low,freq_high,target_s11,tolerance\n')
        except Exception:
            pass
        # 友好版奖励CSV写入
        try:
            friendly_row = [
                ts,
                str(self.global_step),
                str(self.current_step),
                sig,
                f"{reward:.3f}",
                getattr(c, 'name', 'N/A'),
                f"{getattr(c,'freq_low',0.0)/1e9:.3f}",
                f"{getattr(c,'freq_high',0.0)/1e9:.3f}",
                f"{getattr(c,'target_s11',0.0):.2f}",
                f"{getattr(c,'tolerance',0.0):.2f}",
            ]
            with open(self.reward_log_friendly_path, 'a', encoding='utf-8') as f2:
                f2.write(','.join(friendly_row) + '\n')
        except Exception:
            pass
        # 检查终止条件
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        # 获取观测和信息
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
        
    def _connect_hfss(self):
        """连接HFSS"""
        try:
            self.hfss_controller = HFSSController(
                project_path=self.project_path,
                design_name=self.design_name,
                setup_name=self.setup_name,
                sweep_name=self.sweep_name
            )
            self.hfss_controller.connect()
            self.logger.info("HFSS连接成功")
        except Exception as e:
            self.logger.error(f"HFSS连接失败: {e}")
            self.hfss_controller = None
            
    def _update_hfss_model(self) -> bool:
        """更新HFSS模型中的参数向量"""
        # 命中缓存则跳过HFSS更新与仿真
        if self.enable_cache and self._last_param_sig:
            cache_path = self.cache_dir / f"s11_{self._last_param_sig}.csv"
            if cache_path.exists():
                self.last_csv_path = cache_path
                self._used_cache = True
                self.logger.debug("命中仿真缓存，跳过HFSS仿真")
                if self.cache_skip_hfss_update_if_hit:
                    return True
        
        if self.offline:
            return True
        if self.hfss_controller is None:
            return False
            
        try:
            # 直接使用set_variable设置k参数列表（单位mm）
            success = self.hfss_controller.set_variable("k", self.current_params, unit="mm")
            if not success:
                self.logger.error("参数设置失败")
                return False
                
            # 运行仿真
            result = self.hfss_controller.analyze()
            return result
            
        except Exception as e:
            self.logger.error(f"HFSS模型更新失败: {e}")
            return False
            
    def _get_s11_data(self):
        """获取S11数据（CSV优先）"""
        # 若命中缓存，直接读取缓存CSV
        if self._used_cache and self.last_csv_path and self.last_csv_path.exists():
            try:
                data = pd.read_csv(self.last_csv_path)
                freq_col = None
                for c in data.columns:
                    lc = c.lower()
                    if lc.startswith("freq"):
                        freq_col = c
                        break
                if freq_col is None:
                    raise ValueError("CSV缺少频率列")
                freq_vals = data[freq_col].values.astype(float)
                if "ghz" in freq_col.lower():
                    freq_vals = freq_vals * 1e9
                elif np.nanmax(freq_vals) < 1e6:
                    freq_vals = freq_vals * 1e9
                db_cols = [c for c in data.columns if "db(" in c.lower()]
                if len(db_cols) == 0:
                    raise ValueError("CSV缺少S11列")
                s11_vals = data[db_cols[0]].values.astype(float)
                self.frequencies = np.linspace(np.nanmin(freq_vals), np.nanmax(freq_vals), self.freq_samples)
                ok = ~np.isnan(s11_vals) & ~np.isnan(freq_vals)
                self.s11_data = np.interp(self.frequencies, freq_vals[ok], s11_vals[ok])
                return
            except Exception as e:
                self.logger.warning(f"缓存读取失败: {e}; 回退到常规数据获取流程")
        
        # Offline仿真：生成可重复的模拟曲线，便于快速测试
        if self.offline:
            rng = np.random.default_rng(seed=42 + self.current_step)
            if self.frequencies is None:
                fr_low, fr_high = self.sim_freq_range
                self.frequencies = np.linspace(fr_low, fr_high, self.freq_samples)
            center = (self.current_constraint.freq_low + self.current_constraint.freq_high) / 2.0
            bw = (self.current_constraint.freq_high - self.current_constraint.freq_low) / 6.0
            s11 = -5.0 - 15.0 * np.exp(-((self.frequencies - center) ** 2) / (2 * bw * bw))
            noise = rng.normal(0.0, 0.5, size=self.freq_samples)
            self.s11_data = s11 + noise
            
            try:
                cache_path = self.cache_dir / f"s11_{self._last_param_sig}.csv"
                df_cache = pd.DataFrame({
                    'Frequency(Hz)': self.frequencies,
                    'dB(S(1:1,1:1))': self.s11_data
                })
                df_cache.to_csv(cache_path, index=False)
                simple_cache_path = self.cache_dir / f"s11_{self._last_param_sig}_simple.csv"
                pd.DataFrame({
                    'Freq_GHz': np.round(self.frequencies / 1e9, 3),
                    'S11_dB': np.round(self.s11_data, 2)
                }).to_csv(simple_cache_path, index=False)
                self.last_csv_path = cache_path
                self._log_s_params_index(self._last_param_sig, str(cache_path), str(simple_cache_path))
            except Exception:
                pass
            return

        if self.hfss_controller is None:
            return

        try:
            # 获取S参数数据（dB），然后保存CSV并从CSV读取以确保一致性
            df = self.hfss_controller.get_s_params(
                port_combinations=[('1', '1')],
                data_format="dB",
            )

            if df is not None and not df.empty:
                # 直接从内存df解析频率与S11（保留原分辨率）
                try:
                    data = df
                    freq_col = None
                    for c in data.columns:
                        lc = c.lower()
                        if lc.startswith('freq'):
                            freq_col = c
                            break
                    if freq_col is None:
                        freq_col = 'Frequency'
                    freq_vals = data[freq_col].values.astype(float)
                    if 'ghz' in freq_col.lower() or np.nanmax(freq_vals) < 1e6:
                        freq_vals = freq_vals * 1e9
                    db_cols = [c for c in data.columns if 'db(' in c.lower()]
                    if len(db_cols) == 0:
                        s_cols = [c for c in data.columns if c.lower().startswith('s(')]
                        if len(s_cols) == 0:
                            raise ValueError('CSV缺少S11列')
                        raw = data[s_cols[0]].astype(str).values
                        def to_db(v: str) -> float:
                            try:
                                s = v.strip().strip('()')
                                if ',' in s:
                                    a, b = s.split(',', 1)
                                    comp = complex(float(a), float(b))
                                else:
                                    comp = complex(s.replace(' ', ''))
                                mag = np.abs(comp)
                                return 20 * np.log10(mag + 1e-12)
                            except Exception:
                                return np.nan
                        s11_vals = np.array([to_db(v) for v in raw], dtype=float)
                    else:
                        s11_vals = data[db_cols[0]].values.astype(float)
                    # 观测用插值到固定采样点
                    self.frequencies = np.linspace(np.nanmin(freq_vals), np.nanmax(freq_vals), self.freq_samples)
                    ok = ~np.isnan(s11_vals) & ~np.isnan(freq_vals)
                    self.s11_data = np.interp(self.frequencies, freq_vals[ok], s11_vals[ok])
                    # 写入缓存（保留原分辨率）
                    cache_path = self.cache_dir / f's11_{self._last_param_sig}.csv'
                    df_cache = pd.DataFrame({
                        'Frequency(Hz)': freq_vals,
                        'dB(S(1:1,1:1))': s11_vals
                    })
                    df_cache.to_csv(cache_path, index=False)
                    # 简化副本（GHz与固定小数）
                    simple_cache_path = self.cache_dir / f's11_{self._last_param_sig}_simple.csv'
                    df_simple = pd.DataFrame({
                        'Freq_GHz': np.round(freq_vals / 1e9, 3),
                        'S11_dB': np.round(s11_vals, 2)
                    })
                    df_simple.to_csv(simple_cache_path, index=False)
                    self.last_csv_path = cache_path
                    # 索引映射
                    self._log_s_params_index(self._last_param_sig, str(cache_path), str(simple_cache_path))
                except Exception as e:
                    self.logger.warning(f'S11数据解析失败: {e}')
            else:
                self.logger.warning("S11数据获取失败，使用默认值")
                self.s11_data = np.ones(self.freq_samples) * (-5.0)
                if self.frequencies is None:
                    self.frequencies = np.linspace(self.current_constraint.freq_low, self.current_constraint.freq_high, self.freq_samples)

        except Exception as e:
            self.logger.error(f"S11数据获取错误: {e}")
            self.s11_data = np.ones(self.freq_samples) * (-5.0)
            if self.frequencies is None:
                self.frequencies = np.linspace(self.current_constraint.freq_low, self.current_constraint.freq_high, self.freq_samples)
            
    def _extract_physical_features(self):
        """提取物理特征"""
        if self.s11_data is None or self.frequencies is None:
            return
            
        try:
            # 1. 谐振频率（S11最小值对应的频率）
            min_idx = np.argmin(self.s11_data)
            resonant_freq = self.frequencies[min_idx]
            
            # 2. 带宽（-10dB带宽）
            threshold = -10.0
            below_threshold = self.s11_data < threshold
            if np.any(below_threshold):
                indices = np.where(below_threshold)[0]
                bandwidth = self.frequencies[indices[-1]] - self.frequencies[indices[0]]
            else:
                bandwidth = 0.0
                
            # 3. 最小S11值
            min_s11 = np.min(self.s11_data)
            
            # 4. 约束偏差（在约束频段内的平均偏差）
            constraint_mask = (self.frequencies >= self.current_constraint.freq_low) & \
                            (self.frequencies <= self.current_constraint.freq_high)
            if np.any(constraint_mask):
                constraint_s11 = self.s11_data[constraint_mask]
                deviation = np.mean(np.abs(constraint_s11 - self.current_constraint.target_s11))
            else:
                deviation = 100.0  # 大偏差
                
            # 归一化特征
            self.physical_features = np.array([
                resonant_freq / 1e9,  # GHz
                bandwidth / 1e9,      # GHz
                min_s11 / (-50.0),    # 归一化到[-50dB, 0dB]
                deviation / 50.0      # 归一化偏差
            ], dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"物理特征提取错误: {e}")
            self.physical_features = np.zeros(4, dtype=np.float32)
            
    def _calculate_reward(self) -> float:
        """计算基于约束的动态奖励"""
        if self.s11_data is None or self.frequencies is None:
            return -10.0
        
        # 构造频段掩码（支持约束组）
        bands = []
        if self.current_constraint_group is not None:
            bands = self.current_constraint_group.constraints
        else:
            bands = [self.current_constraint]
        
        # 联合带外掩码
        mask_union = np.zeros_like(self.frequencies, dtype=bool)
        inband_reward = 0.0
        performance_reward = 0.0
        total_weight = 0.0
        
        for c in bands:
            band_mask = (self.frequencies >= c.freq_low) & (self.frequencies <= c.freq_high)
            if not np.any(band_mask):
                # 无有效频段，给予小惩罚但继续
                performance_reward -= 1.0
                continue
            mask_union |= band_mask
            s11_band = self.s11_data[band_mask]
            target = c.target_s11
            tol = c.tolerance
            w = getattr(c, 'weight', 1.0)
            total_weight += w
            
            # 满足度: 不等式满足比例 (S11 <= target + tol)
            satisfied = np.sum(s11_band <= (target + tol))
            satisfaction_ratio = satisfied / len(s11_band)
            inband_reward += w * (satisfaction_ratio * 10.0)
            
            # 违背程度: 超过目标的平均超量 (S11 - target 的正部)
            excess = np.maximum(0.0, s11_band - target)
            mean_excess = float(np.mean(excess))
            performance_reward += w * max(0.0, 10.0 - mean_excess)
        
        if total_weight > 0:
            inband_reward /= total_weight
            performance_reward /= total_weight
        
        # 带外惩罚：非任何频段的点，如果S11低于阈值则惩罚
        out_mask = ~mask_union
        out_penalty = 0.0
        if np.any(out_mask):
            out_s11 = self.s11_data[out_mask]
            bad_points = np.sum(out_s11 < self.out_of_band_threshold)
            out_penalty = bad_points * (-0.5)
        
        # 物理合理性奖励
        physics_reward = 0.0
        if self.physical_features[1] > 0:  # 有带宽
            physics_reward += 1.0
        if self.physical_features[2] > 0.2:  # 有一定的S11深度
            physics_reward += 1.0
            
        # 总奖励
        total_reward = inband_reward + performance_reward + out_penalty + physics_reward
        
        return float(total_reward)
        
    def _is_terminated(self) -> bool:
        """检查是否达到终止条件"""
        if self.s11_data is None or self.frequencies is None:
            return False
        
        bands = []
        if self.current_constraint_group is not None:
            bands = self.current_constraint_group.constraints
        else:
            bands = [self.current_constraint]
        
        # 所有频段均满足则终止
        for c in bands:
            band_mask = (self.frequencies >= c.freq_low) & (self.frequencies <= c.freq_high)
            if not np.any(band_mask):
                return False
            s11_band = self.s11_data[band_mask]
            if not np.all(s11_band <= (c.target_s11 + c.tolerance)):
                return False
        return True
        
    def _get_observation(self) -> np.ndarray:
        """获取当前观测"""
        # 参数向量
        param_obs = self.current_params.astype(np.float32)
        
        # S11数据
        s11_obs = self.s11_data if self.s11_data is not None else np.ones(self.freq_samples) * (-5.0)
        s11_obs = s11_obs.astype(np.float32)
        
        # 物理特征
        physics_obs = self.physical_features.astype(np.float32)
        
        # 约束向量：按max_bands填充 [fl, fh, target] * max_bands + [oob_threshold]
        constraint_vec = []
        if self.current_constraint_group is not None:
            bands = self.current_constraint_group.constraints
        else:
            bands = [self.current_constraint]
        
        for i in range(self.max_bands):
            if i < len(bands):
                c = bands[i]
                constraint_vec.extend([
                    c.freq_low / 1e9,
                    c.freq_high / 1e9,
                    c.target_s11 / (-50.0)
                ])
            else:
                constraint_vec.extend([0.0, 0.0, 0.0])
        # 带外阈值（归一化到[-50,0]）
        constraint_vec.append(self.out_of_band_threshold / (-50.0))
        constraint_obs = np.array(constraint_vec, dtype=np.float32)
        
        observation = np.concatenate([param_obs, s11_obs, physics_obs, constraint_obs])
        return observation
        
    def _get_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        info = {
            'step': self.current_step,
            'constraint': {
                'freq_low': self.current_constraint.freq_low,
                'freq_high': self.current_constraint.freq_high,
                'target_s11': self.current_constraint.target_s11
            },
            'physical_features': {
                'resonant_freq': self.physical_features[0] * 1e9 if len(self.physical_features) > 0 else 0,
                'bandwidth': self.physical_features[1] * 1e9 if len(self.physical_features) > 1 else 0,
                'min_s11': self.physical_features[2] * (-50.0) if len(self.physical_features) > 2 else 0,
                'deviation': self.physical_features[3] * 50.0 if len(self.physical_features) > 3 else 0
            },
            'param_sum_mm': float(np.sum(self.current_params)),
            's11_available': self.s11_data is not None
        }
        if self.current_constraint_group is not None:
            info['constraint_group'] = {
                'name': self.current_constraint_group.name,
                'n_bands': len(self.current_constraint_group.constraints)
            }
        return info
        
    def render(self, mode='human'):
        """渲染环境（可选实现）"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Params: {np.sum(self.current_params)}/{self.param_count}")
            if self.current_constraint_group is not None:
                print(f"ConstraintGroup: {self.current_constraint_group.name}, Bands={len(self.current_constraint_group.constraints)}, OOB>{self.out_of_band_threshold}dB")
            else:
                print(f"Constraint: {self.current_constraint.freq_low/1e9:.1f}-{self.current_constraint.freq_high/1e9:.1f}GHz, Target: {self.current_constraint.target_s11}dB")
            if self.s11_data is not None:
                print(f"S11 range: [{np.min(self.s11_data):.1f}, {np.max(self.s11_data):.1f}]dB")
    def _compute_param_signature(self) -> str:
        vals = [float(v) for v in np.asarray(self.current_params).tolist()]
        # 固定量化，避免微小浮点噪声导致签名不同
        quantized = [round(v, 6) for v in vals]
        payload = {
            'mode': getattr(self, 'action_mode', 'discrete'),
            'bounds': list(self.variable_bounds),
            'params': quantized
        }
        s = json.dumps(payload, separators=(',', ':'), ensure_ascii=False)
        return hashlib.sha1(s.encode('utf-8')).hexdigest()

    def _log_current_params(self, sig: str):
        try:
            constraint_name = getattr(self.current_constraint, 'name', 'N/A')
            fr = f"{self.current_constraint.freq_low:.3e}-{self.current_constraint.freq_high:.3e}"
            ts = datetime.now().isoformat(timespec='seconds')
            row = [ts, str(self.current_step), getattr(self, 'action_mode', 'discrete'), sig,
                   json.dumps([float(v) for v in self.current_params.tolist()]),
                   constraint_name, fr, str(self.current_constraint.target_s11), str(self.current_constraint.tolerance)]
            with open(self.param_log_path, 'a', encoding='utf-8') as f:
                f.write(','.join(row) + '\n')
        except Exception:
            pass

    def _log_step_reward(self, reward: float):
        ts = datetime.now().isoformat(timespec='seconds')
        sig = self._last_param_sig or ''
        c = self.current_constraint
        row = [
            ts,
            str(self.current_step),
            sig,
            f"{reward:.6f}",
            getattr(c, 'name', 'N/A'),
            f"{getattr(c,'freq_low',0.0):.6e}",
            f"{getattr(c,'freq_high',0.0):.6e}",
            f"{getattr(c,'target_s11',0.0)}",
            f"{getattr(c,'tolerance',0.0)}",
        ]
        with open(self.reward_log_path, 'a', encoding='utf-8') as f:
            f.write(','.join(row) + '\n')

    def _setup_spaces(self):
        """初始化观察与动作空间。"""
        # 观测维度 = 参数向量 + 频谱采样(self.freq_samples) + 物理特征(4) + 约束向量
        self.observation_dim = int(self.param_count + self.freq_samples + 4 + self.constraint_vector_dim)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32)
        
        # 动作空间：连续或离散
        mode = str(self.action_mode).lower()
        if mode == 'continuous':
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.param_count,), dtype=np.float32)
        elif mode == 'binary':
            self.action_space = spaces.MultiBinary(self.param_count)
        elif mode in ('list', 'multidiscrete') and (self.allowed_param_values is not None):
            self.action_space = spaces.MultiDiscrete([len(self.allowed_param_values)] * self.param_count)
        else:
            # 旧的离散模式：选择某个参数索引进行翻转
            self.action_space = spaces.Discrete(self.param_count)

        # 全局步计数（用于易读奖励日志与索引）
        if not hasattr(self, 'global_step'):
            self.global_step = 0
        
        # 频谱采样默认
        if self.frequencies is None:
            fr_low, fr_high = self.sim_freq_range
            self.frequencies = np.linspace(fr_low, fr_high, self.freq_samples)

    def _log_s_params_index(self, param_sig: str, cache_path: str, simple_path: str):
        """记录参数签名到S参数缓存文件路径的索引映射。"""
        try:
            timestamp = datetime.now().isoformat(timespec='seconds')
            constraint_name = getattr(self.current_constraint, 'name', '') if hasattr(self, 'current_constraint') else ''
            row = f"{timestamp},{int(self.global_step)},{param_sig},{constraint_name},{cache_path},{simple_path}\n"
            with open(self.sparams_index_path, 'a', encoding='utf-8') as f:
                f.write(row)
        except Exception:
            pass