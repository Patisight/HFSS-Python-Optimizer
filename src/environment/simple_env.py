"""
简化的天线环境

这是一个简化版本的天线优化环境，去除了复杂的约束管理和参数化功能，
只保留基本的强化学习环境接口。
"""

import numpy as np
import gym
from gym import spaces
import logging
from typing import Tuple, Dict, Any, Optional
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api import AntennaAPI

logger = logging.getLogger(__name__)

class SimpleAntennaEnv(gym.Env):
    """
    简化的天线优化环境
    
    状态空间: 当前天线参数 + S参数特征
    动作空间: 天线参数的调整量
    奖励函数: 基于S11参数的简单奖励
    """
    
    def __init__(self):
        super().__init__()
        
        # 初始化HFSS API
        self.api = AntennaAPI()
        
        # 定义动作和观测空间
        # 动作空间: 4个天线参数的调整量 [-0.1, 0.1]
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(4,), dtype=np.float32
        )
        
        # 观测空间: 4个天线参数 + 简化的S参数特征
        # 天线参数范围 [0, 20] + S11特征 [-50, 0]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -50, -50, -50]),
            high=np.array([20, 20, 20, 20, 0, 0, 0]),
            dtype=np.float32
        )
        
        # 环境参数
        self.param_bounds = {
            'l1': (1.0, 20.0),
            'l2': (1.0, 20.0), 
            'w1': (0.5, 10.0),
            'w2': (0.5, 10.0)
        }
        
        # 目标频率和S11
        self.target_freq = 2.4e9  # 2.4GHz
        self.target_s11 = -10.0   # -10dB
        
        # 当前状态
        self.current_params = None
        self.step_count = 0
        self.max_steps = 50
        
        logger.info("简化天线环境初始化完成")
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        # 随机初始化天线参数
        self.current_params = {
            'l1': np.random.uniform(5.0, 15.0),
            'l2': np.random.uniform(5.0, 15.0),
            'w1': np.random.uniform(2.0, 8.0),
            'w2': np.random.uniform(2.0, 8.0)
        }
        
        self.step_count = 0
        
        # 获取初始观测
        observation = self._get_observation()
        
        logger.debug(f"环境重置，初始参数: {self.current_params}")
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行一步动作"""
        self.step_count += 1
        
        # 应用动作到参数
        param_names = ['l1', 'l2', 'w1', 'w2']
        for i, param_name in enumerate(param_names):
            # 应用动作
            new_value = self.current_params[param_name] + action[i]
            
            # 限制在边界内
            min_val, max_val = self.param_bounds[param_name]
            self.current_params[param_name] = np.clip(new_value, min_val, max_val)
        
        # 获取新的观测和奖励
        observation = self._get_observation()
        reward = self._calculate_reward()
        
        # 检查是否结束
        done = self.step_count >= self.max_steps
        
        info = {
            'step': self.step_count,
            'params': self.current_params.copy(),
            'reward_components': self._get_reward_components()
        }
        
        return observation, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观测"""
        try:
            # 获取S参数
            s_params = self.api.get_s_parameters(self.current_params)
            
            # 提取关键频率点的S11值
            freqs = s_params['frequencies']
            s11_db = s_params['s11_db']
            
            # 找到目标频率附近的S11值
            target_idx = np.argmin(np.abs(freqs - self.target_freq))
            s11_at_target = s11_db[target_idx]
            
            # 计算简单的S11统计特征
            s11_mean = np.mean(s11_db)
            s11_min = np.min(s11_db)
            
            # 构建观测向量
            observation = np.array([
                self.current_params['l1'],
                self.current_params['l2'],
                self.current_params['w1'],
                self.current_params['w2'],
                s11_at_target,
                s11_mean,
                s11_min
            ], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"获取S参数失败: {e}")
            # 返回默认观测
            observation = np.array([
                self.current_params['l1'],
                self.current_params['l2'],
                self.current_params['w1'],
                self.current_params['w2'],
                -5.0,  # 默认S11值
                -5.0,
                -5.0
            ], dtype=np.float32)
        
        return observation
    
    def _calculate_reward(self) -> float:
        """计算奖励"""
        try:
            # 获取S参数
            s_params = self.api.get_s_parameters(self.current_params)
            freqs = s_params['frequencies']
            s11_db = s_params['s11_db']
            
            # 找到目标频率的S11值
            target_idx = np.argmin(np.abs(freqs - self.target_freq))
            s11_at_target = s11_db[target_idx]
            
            # 基于S11的简单奖励函数
            # 目标是S11 < -10dB，越小越好
            if s11_at_target < self.target_s11:
                # 达到目标，给予正奖励
                reward = 1.0 + (self.target_s11 - s11_at_target) * 0.1
            else:
                # 未达到目标，给予负奖励
                reward = -1.0 - (s11_at_target - self.target_s11) * 0.1
            
            # 限制奖励范围
            reward = np.clip(reward, -10.0, 10.0)
            
        except Exception as e:
            logger.warning(f"计算奖励失败: {e}")
            reward = -1.0  # 默认负奖励
        
        return float(reward)
    
    def _get_reward_components(self) -> Dict[str, float]:
        """获取奖励组成部分（用于调试）"""
        try:
            s_params = self.api.get_s_parameters(self.current_params)
            freqs = s_params['frequencies']
            s11_db = s_params['s11_db']
            
            target_idx = np.argmin(np.abs(freqs - self.target_freq))
            s11_at_target = s11_db[target_idx]
            
            return {
                's11_at_target': s11_at_target,
                'target_s11': self.target_s11,
                'meets_target': s11_at_target < self.target_s11
            }
        except:
            return {'error': True}
    
    def render(self, mode='human'):
        """渲染环境（可选实现）"""
        if mode == 'human':
            print(f"Step: {self.step_count}")
            print(f"Parameters: {self.current_params}")
            reward_info = self._get_reward_components()
            if 'error' not in reward_info:
                print(f"S11 at target: {reward_info['s11_at_target']:.2f} dB")
    
    def close(self):
        """关闭环境"""
        if hasattr(self.api, 'close'):
            self.api.close()
        logger.info("环境已关闭")