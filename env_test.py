import numpy as np
# 假设你已经正确导入并定义了 ElectricVehicleEnv 环境
from env_new import ElectricVehicleEnv


def test_environment(env, num_steps=10, seed=42):
    # 设置随机种子
    np.random.seed(seed)
    env.action_space.seed(seed)

    # 重置环境以开始新的回合
    state = env.reset()
    done = False

    print("Initial State:")
    print_state(state)
    #

    # 进行一系列的随机动作，并打印每个step后的三要素
    for step in range(num_steps):
        if done:
            print("Episode finished!")
            break

        # 随机选择一个动作
        action = env.action_space.sample()

        # 执行该动作，并获取结果
        state, reward, done, info = env.step(action)
        if not done:
            # 打印当前的状态、奖励、以及是否完成
            print_step_details(info)
        print(f"\nStep {step + 1}:")

        print_state(state)
        print("Reward:", reward)
        print("Done:", done)
        print("-" * 30)


def print_state(state):
    """
    打印状态的每个属性，并标注其含义。
    """
    print(f"  当前节点 (Current Node): {state[0]}")
    print(f"  当前电池电量 (Battery): {state[1]}")
    print(f"  剩余时间 (Remaining Time): {state[2]}")
    print(f"  目标节点 (Target Node): {state[3]}")


def print_step_details(info):
    """
    打印每个步骤的详细信息。
    """
    print(f"  从节点 (From Node): {info['from_node']} 到节点 (To Node): {info['to_node']}")
    print(f"  距离 (Distance): {info['distance']} km")
    print(f"  耗电 (Energy Consumed): {info['energy_consumed']} kWh")
    print(f"  是否充电 (Charging): {'Yes' if info['charging'] else 'No'}")
    if info['charging']:
        print(f"  充电量 (Energy Charged): {info['energy_charged']} kWh")
    print(f"  电量从 (Battery Start): {info['battery_start']} kWh 变成 (Battery End): {info['battery_end']} kWh")
    print(f"  剩余时间从 (Time Start): {info['time_start']} h 变成 (Time End): {info['time_end']} h")


# 创建环境实例
env = ElectricVehicleEnv()

# 测试环境，执行10步随机动作
test_environment(env, num_steps=10, seed=42)
