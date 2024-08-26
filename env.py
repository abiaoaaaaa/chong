import gym
from gym import spaces
import numpy as np


class FullyConnectedGraphEnv(gym.Env):
    def __init__(self, num_nodes, edge_weights, max_steps=30):
        super(FullyConnectedGraphEnv, self).__init__()
        self.num_nodes = num_nodes
        self.edge_weights = edge_weights
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.MultiDiscrete([num_nodes, num_nodes])
        self.current_node = None
        self.target_node = None
        self.step_count = 0  # 添加步骤计数器

    def reset(self, start_node=None, end_node=None):
        self.current_node = np.random.randint(self.num_nodes) if start_node is None else start_node
        self.target_node = np.random.randint(self.num_nodes) if end_node is None else end_node
        self.step_count = 0  # 重置步骤计数器
        return np.array([self.current_node, self.target_node])

    def step(self, action):
        next_node = action
        reward = -self.edge_weights[self.current_node][next_node]
        self.current_node = next_node
        self.step_count += 1  # 增加步骤计数器

        done = self.current_node == self.target_node or self.step_count >= self.max_steps
        if done:
            if self.current_node == self.target_node:
                reward += 100  # 到达目标节点的奖励
            else:
                reward -= 10  # 超过最大步数的惩罚

        return np.array([self.current_node, self.target_node]), reward, done, {}

    def render(self, mode='human'):
        print(f'Current node: {self.current_node}, Target node: {self.target_node}, Steps: {self.step_count}')
