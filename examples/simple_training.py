"""
简化的强化学习训练示例

这是一个最基本的训练示例，展示如何使用简化的强化学习框架进行天线优化训练。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import logging
from pathlib import Path

from src.environment.simple_env import SimpleAntennaEnv
from src.agent.simple_agent import SimplePPOAgent
from src.training.simple_trainer import SimpleTrainer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """主训练函数"""
    
    # 创建环境
    env = SimpleAntennaEnv()
    
    # 创建智能体
    agent = SimplePPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    
    # 创建训练器
    trainer = SimpleTrainer(env, agent)
    
    # 开始训练
    logger.info("开始训练...")
    results = trainer.train(max_episodes=100)
    
    # 保存模型
    model_path = "models/simple_model.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    agent.save(model_path)
    
    logger.info(f"训练完成！模型已保存到: {model_path}")
    logger.info(f"最终平均奖励: {results['final_avg_reward']:.4f}")

if __name__ == "__main__":
    main()