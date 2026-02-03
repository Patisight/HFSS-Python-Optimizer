"""
简化的训练器

这是一个简化版本的训练器，去除了复杂的课程学习、约束管理等功能，
只保留基本的强化学习训练循环。
"""

import numpy as np
import logging
from typing import Dict, Any
from collections import deque
import time

logger = logging.getLogger(__name__)

class SimpleTrainer:
    """
    简化的强化学习训练器
    
    实现基本的训练循环，包括环境交互、经验收集和智能体更新。
    """
    
    def __init__(self, env, agent, update_frequency: int = 10):
        """
        初始化训练器
        
        Args:
            env: 环境实例
            agent: 智能体实例
            update_frequency: 更新频率（每多少个episode更新一次）
        """
        self.env = env
        self.agent = agent
        self.update_frequency = update_frequency
        
        # 训练统计
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_stats = {
            'total_episodes': 0,
            'total_steps': 0,
            'best_reward': float('-inf'),
            'policy_losses': [],
            'value_losses': []
        }
        
        logger.info("简化训练器初始化完成")
    
    def train(self, max_episodes: int = 100, log_frequency: int = 10) -> Dict[str, Any]:
        """
        开始训练
        
        Args:
            max_episodes: 最大训练episode数
            log_frequency: 日志输出频率
            
        Returns:
            训练结果统计
        """
        logger.info(f"开始训练，最大episode数: {max_episodes}")
        start_time = time.time()
        
        for episode in range(max_episodes):
            # 运行一个episode
            episode_result = self._run_episode()
            
            # 更新统计
            self.episode_rewards.append(episode_result['reward'])
            self.episode_lengths.append(episode_result['length'])
            self.training_stats['total_episodes'] += 1
            self.training_stats['total_steps'] += episode_result['length']
            
            # 更新最佳奖励
            if episode_result['reward'] > self.training_stats['best_reward']:
                self.training_stats['best_reward'] = episode_result['reward']
            
            # 定期更新智能体
            if (episode + 1) % self.update_frequency == 0:
                update_result = self.agent.update()
                if update_result:
                    self.training_stats['policy_losses'].append(update_result.get('policy_loss', 0))
                    self.training_stats['value_losses'].append(update_result.get('value_loss', 0))
            
            # 定期输出日志
            if (episode + 1) % log_frequency == 0:
                self._log_progress(episode + 1)
        
        # 训练完成
        training_time = time.time() - start_time
        logger.info(f"训练完成！总用时: {training_time:.2f}秒")
        
        return self._get_training_summary(training_time)
    
    def _run_episode(self) -> Dict[str, Any]:
        """运行一个episode"""
        state = self.env.reset()
        total_reward = 0
        step_count = 0
        done = False
        
        while not done:
            # 选择动作
            action, log_prob, value = self.agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 存储经验
            self.agent.store_experience(
                state, action, reward, next_state, done, log_prob, value
            )
            
            # 更新状态和统计
            state = next_state
            total_reward += reward
            step_count += 1
            
            # 防止无限循环
            if step_count >= 200:
                done = True
        
        return {
            'reward': total_reward,
            'length': step_count
        }
    
    def _log_progress(self, episode: int):
        """输出训练进度"""
        if len(self.episode_rewards) == 0:
            return
        
        avg_reward = np.mean(self.episode_rewards)
        avg_length = np.mean(self.episode_lengths)
        
        logger.info(
            f"Episode {episode}: "
            f"平均奖励 = {avg_reward:.4f}, "
            f"平均长度 = {avg_length:.1f}, "
            f"最佳奖励 = {self.training_stats['best_reward']:.4f}"
        )
        
        # 输出损失信息
        if self.training_stats['policy_losses']:
            recent_policy_loss = np.mean(self.training_stats['policy_losses'][-10:])
            recent_value_loss = np.mean(self.training_stats['value_losses'][-10:])
            logger.info(
                f"最近损失: 策略损失 = {recent_policy_loss:.6f}, "
                f"价值损失 = {recent_value_loss:.6f}"
            )
    
    def _get_training_summary(self, training_time: float) -> Dict[str, Any]:
        """获取训练总结"""
        if len(self.episode_rewards) == 0:
            final_avg_reward = 0
        else:
            final_avg_reward = np.mean(self.episode_rewards)
        
        summary = {
            'training_time': training_time,
            'total_episodes': self.training_stats['total_episodes'],
            'total_steps': self.training_stats['total_steps'],
            'final_avg_reward': final_avg_reward,
            'best_reward': self.training_stats['best_reward'],
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0
        }
        
        if self.training_stats['policy_losses']:
            summary['final_policy_loss'] = np.mean(self.training_stats['policy_losses'][-10:])
            summary['final_value_loss'] = np.mean(self.training_stats['value_losses'][-10:])
        
        return summary
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        评估智能体性能
        
        Args:
            num_episodes: 评估episode数
            
        Returns:
            评估结果
        """
        logger.info(f"开始评估，episode数: {num_episodes}")
        
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            step_count = 0
            done = False
            
            while not done:
                # 使用确定性策略
                action, _, _ = self.agent.select_action(state, deterministic=True)
                next_state, reward, done, info = self.env.step(action)
                
                state = next_state
                total_reward += reward
                step_count += 1
                
                if step_count >= 200:
                    done = True
            
            eval_rewards.append(total_reward)
            eval_lengths.append(step_count)
        
        eval_result = {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards),
            'avg_length': np.mean(eval_lengths)
        }
        
        logger.info(
            f"评估完成: 平均奖励 = {eval_result['avg_reward']:.4f} ± {eval_result['std_reward']:.4f}"
        )
        
        return eval_result