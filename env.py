import numpy as np
import pandas as pd
# 函数块 1
# 导入load_and_process_data函数
from data_lode import load_and_process_data
import tensorflow as tf
#
import gymnasium as gym
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import fsolve
import os
warnings.filterwarnings("ignore")
#np.random.seed(120)
tf.random.set_seed(120)
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from stable_baselines3 import DDPG

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
class ElectricVehicleEnv(gym.Env):
    def __init__(self):
        super(ElectricVehicleEnv, self).__init__()
        #下载数据
        data_directory = 'data'
        processed_data = load_and_process_data(data_directory)
        # 访问各个DataFrame
        self.data_50 = processed_data['data_50']
        self.evs_50 = processed_data['evs_50']
        self.distance_50 = processed_data['distance_50']
        self.roads_50 = processed_data['roads_50']
        self.speed_50 = processed_data['speed_50']
        # 调用函数并计算时间
        self.time_50 = self.calculate_travel_time()
        # 环境的初始设置
        self.t = 0
        self.num_nodes = len(self.data_50)
        self.num_vehicles = len(self.evs_50)
        self.current_vehicle = 0
        self.current_node = None
        self.current_time = 0
        self.current_battery = None
        self.target_node = None
        self.remaining_time = None
        self.done = False
        # 定义动作空间和观察空间
        self.action_space = gym.spaces.Discrete(self.num_nodes)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array(
                [self.num_nodes - 1, 50, 50, self.num_nodes - 1],
                dtype=np.float32),
            dtype=np.float32
        )
        self.normalize = np.array([1, 0.5 , 25 , 1])
        self.truncated = False
        # 初始化车辆参数
        self.reset()

    def reset(self,seed=0):
        """
        环境重置，初始化当前车辆的状态
        """
        # 随机选择一辆车作为当前模拟对象
        self.current_vehicle = np.random.choice(self.num_vehicles)
        #使用一个数据,5个随机数
        self.current_vehicle = np.random.randint(0, 10)
        vehicle_data = self.evs_50.iloc[self.current_vehicle]

        # 设定起点、终点、初始电量、截止时间等信息
        self.current_node = vehicle_data['起点编号']
        self.target_node = vehicle_data['终点编号']
        self.current_battery = vehicle_data['初始电量']
        self.remaining_time = vehicle_data['截止时间']
        self.remaining_time_initial = self.remaining_time
        self.current_time = 0

        self.done = False

        self.t = 0
        # 初始化 info 字典
        self.info = {}
        self.truncated = False
        return self._get_state(), self.info


    def step(self, action):
        """
        执行一步行动，即从当前位置移动到指定的下一个位置
        参数:
        action (int): 下一个要移动到的顶点编号
        返回:
        state (dict): 移动后的新状态
        self.reward (float): 执行该动作获得的奖励
        done (bool): 当前车辆是否已到达终点或任务结束
        """
        self.t += 1
        # 获取动作目标节点的距离和时间
        next_node = int(action)
        self.current_node = int(self.current_node)
        self.target_node = int(self.target_node)

        # 检查动作的有效性
        #if self.distance_50.iloc[self.current_node, next_node] == 0:
            #self.done = True
            #return self._get_state(), -100, self.done, self.truncated, self.info

        distance = self.distance_50.iloc[self.current_node, next_node]

        travel_time = self.time_50.iloc[self.current_node, next_node]
        energy_consumed = distance * self.evs_50.iloc[self.current_vehicle]['行驶能耗'] / 100

        battery_start = self.current_battery
        time_start = self.remaining_time
        distance_to_target_start = self.distance_50.iloc[self.current_node, self.target_node]
        # 更新状态
        self.last_node = self.current_node
        self.current_node = next_node
        self.current_battery -= energy_consumed

        self.current_time += travel_time
        self.remaining_time -= travel_time
        charging = False
        energy_charged = 0


        # 计算距离目标节点的变化

        distance_to_target_end = self.distance_50.iloc[self.current_node, self.target_node]
        distance_decrease = distance_to_target_start - distance_to_target_end



        distance_reward = distance_decrease         # 距离接近终点越多，奖励越大
        #计算奖励和判断是否完成任务
        r0 = -energy_consumed
        r1 = distance_reward * self.evs_50.iloc[self.current_vehicle]['行驶能耗'] / 100   # 接近目标距离转换成电量的正奖励
        r2 = 0
        if self.roads_50.iloc[self.last_node, next_node] == 1:  # 判断是否在充电路段
            charging_power = 100  # 假设充电功率为100kW
            #最多能充多少电
            energy_charged = min(charging_power * travel_time,
                                 self.evs_50.iloc[self.current_vehicle]['电池容量'] - self.current_battery)
            self.current_battery += energy_charged
            charging = True
            time_factor = max(0, (self.remaining_time / self.remaining_time_initial))
            charging_reward = time_factor * energy_charged  # 根据时间剩余动态调整充电奖励
            #调试，先不给他充电奖励
            r2 = charging_reward  # 充电的正奖励
        self.reward = 0
        self.reward = r0 * 10 + r1 * 10 + r2*10 - 100

        if self.current_node == self.target_node:  # 到终点
            self.reward += self.current_battery
            self.done = True
        elif self.remaining_time <= 0 or self.current_battery <= 0 or self.t > 10 or self.last_node == self.current_node:
            self.done = True
            self.reward -= 8000   # 任务失败的惩罚


            # 额外信息，用于调试
        self.info = {
                'from_node': self.last_node,
                'to_node': self.current_node,
                'distance_decrease': distance_decrease,
                'energy_consumed': energy_consumed,
                'charging': charging,
                'energy_charged': energy_charged,
                'battery_start': battery_start,
                'battery_end': self.current_battery,
                'time_start': time_start,
                'time_end': self.remaining_time
            }
        return self._get_state(), self.reward, self.done, self.truncated, self.info


    def render(self, mode='human'):
        """
        可视化当前环境状态（此处为简单打印状态）
        """
        print(f"Current Vehicle: {self.current_vehicle}")
        print(f"Current Node: {self.current_node}")
        print(f"Battery: {self.current_battery}")
        print(f"Remaining Time: {self.remaining_time}")
        print(f"Target Node: {self.target_node}")
        print(f"Done: {self.done}")

    def _get_state(self):
        """
        获取当前状态，包含当前位置、电量、剩余时间,目标点
        """
        state = np.array([
            self.current_node,  # 当前节点
            self.current_battery,  # 当前电池电量
            self.remaining_time,  # 剩余时间
            self.target_node  # 目标节点
        ], dtype=np.float32)
        for i in range(4):
            state[i] = state[i] * self.normalize[i]
        return state
    def calculate_travel_time(self):
        distance_50 = pd.read_csv('data/distance_50.csv', header=None)
        speed_50 = pd.read_csv('data/speed_50.csv', header=None)
        """
        根据距离和速度计算每条路所需的时间。
        """
        # 计算时间 = 距离 / 速度
        time_50 = distance_50 / speed_50
        time_50 = time_50.fillna(0)
        return time_50

# 示例使用
# 假设你已经读取了数据集，并保存为如下变量
# data_50, evs_50, distance_50, roads_50, speed_50, time_50

# env = ElectricVehicleEnv(data_50, evs_50, distance_50, roads_50, speed_50, time_50)
# state = env.reset()
# done = False
# while not done:
#     action = np.random.choice(env.num_nodes)  # 这里随意选择一个动作，实际应用中需用策略选择
#     state, self.reward, done = env.step(action)
#     env.render()
