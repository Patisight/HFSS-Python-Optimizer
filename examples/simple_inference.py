"""
简化的强化学习推理示例

这是一个最基本的推理示例，展示如何使用训练好的模型进行天线参数推理。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import logging

from src.environment.simple_env import SimpleAntennaEnv
from src.agent.simple_agent import SimplePPOAgent

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """主推理函数"""
    
    # 创建环境
    env = SimpleAntennaEnv()
    
    # 创建智能体
    agent = SimplePPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    
    # 加载训练好的模型
    model_path = "models/simple_model.pth"
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        logger.info("请先运行 simple_training.py 进行训练")
        return
    
    agent.load(model_path)
    logger.info(f"模型已加载: {model_path}")
    
    # 进行推理测试
    logger.info("开始推理测试...")
    
    total_reward = 0
    num_episodes = 10
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 100:
            # 使用训练好的策略选择动作
            action = agent.select_action(state, deterministic=True)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            step += 1
        
        total_reward += episode_reward
        logger.info(f"Episode {episode + 1}: 奖励 = {episode_reward:.4f}, 步数 = {step}")
    
    avg_reward = total_reward / num_episodes
    logger.info(f"推理完成！平均奖励: {avg_reward:.4f}")

if __name__ == "__main__":
    main()