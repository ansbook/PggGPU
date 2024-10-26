import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time
# 检查可用的 GPU 设备数量
cp.cuda.Device(0).use()
#计算整个运行时间
start_time = time.time()
class Lattice_Game():
    def __init__(self, n, wealthy, r, p=[0.5,0.5], strategy=[0,1]):
        # 策略矩阵，数量n，策略数组，概率数组
        self.strategy_matrix = cp.random.choice(strategy, size=(n,n), p=p)
        self.wealthy_matrix = cp.full((n,n), wealthy, dtype=cp.float64)
        self.r = r
        self.n = n
        self.step_count = 0  # 初始化步数计数器
        self.cooperation_trend = []  # 用于存储合作者比例的列表
        cooperation_ratio = self.get_strategy()/(self.n*self.n)  # 计算当前合作者比例
        self.cooperation_trend.append(cooperation_ratio)  # 将当前合作者比例加入列表
        # print("初始矩阵")
        # print(self.strategy_matrix)
        # print(self.wealthy_matrix)
        # print("------")
    def game(self):
        self.wealthy_matrix = self.wealthy_matrix - 5 * self.strategy_matrix

    def create_payoff_matrix(self, original_matrix):
        # 类似卷积的操作，先移动上下左右，然后累加求和,这样就得到了一次博弈的收益矩阵
        top = cp.roll(original_matrix, 1, axis=0)
        bottom = cp.roll(original_matrix, -1, axis=0)
        left = cp.roll(original_matrix, 1, axis=1)
        right = cp.roll(original_matrix, -1, axis=1)
        new_matrix = top + bottom + left + right + original_matrix
        return new_matrix/5
    def payoff(self):

        center = self.strategy_matrix
        top = cp.roll(self.strategy_matrix, 1, axis=0)
        bottom = cp.roll(self.strategy_matrix, -1, axis=0)
        left = cp.roll(self.strategy_matrix, 1, axis=1)
        right = cp.roll(self.strategy_matrix, -1, axis=1)
        self.wealthy_matrix = self.wealthy_matrix + self.r * (self.create_payoff_matrix(center)
                                                              + self.create_payoff_matrix(top)
                                                              + self.create_payoff_matrix(bottom)
                                                              + self.create_payoff_matrix(left)
                                                              + self.create_payoff_matrix(right))
        # print("博弈后")
        # print(self.wealthy_matrix)
        # print("------")

    #gpu实现
    def change_strategy(self, beta=0.01):
        n = self.n
        directions = [(1,0),(-1,0),(1,1),(-1,1)] #随机选取一个方向
        #directions = [(1,0)] #只选取一个方向
        direction_indices = np.random.choice(len(directions))
        dx, dy = directions[direction_indices]

        # 随机选取一个方向进行整体更新
        top = cp.roll(self.wealthy_matrix, dx, axis=dy)
        top_strategy = cp.roll(self.strategy_matrix, dx, axis=dy)
        # 计算费米函数
        cdf = 1/(1 + cp.exp(self.wealthy_matrix - top) / beta)

        # 根据费米更新原则更新策略
        random_matrix = cp.random.rand(n, n)

        update_mask = random_matrix < cdf

        #更新
        self.strategy_matrix = cp.where(update_mask, top_strategy, self.strategy_matrix)

    def plot_simulation(self):
        strategy_np = self.strategy_matrix.get()
        cooperators = np.argwhere(strategy_np == 1)  # 找到合作者的坐标
        cheaters = np.argwhere(strategy_np == 0)  # 找到背叛者的坐标

        plt.scatter(cooperators[:, 1], cooperators[:, 0], c='green')  # 绘制合作者，绿色
        plt.scatter(cheaters[:, 1], cheaters[:, 0], c='red')  # 绘制背叛者，红色

        plt.title("Monte Carlo Simulation")
        plt.legend()
        plt.show()

    def plot_cooperation_trend(self):
        cooperation_trend_np = cp.asnumpy(cp.array(self.cooperation_trend))
        plt.plot(range(len(cooperation_trend_np)), cooperation_trend_np)
        plt.title("Cooperation Trend")
        plt.xlabel("Step")
        plt.ylabel("Cooperation Ratio")
        plt.show()
    def step(self):
        self.game()
        self.payoff()
        self.change_strategy()
        self.step_count += 1  # 每执行一步，步数计数器加1
        cooperation_ratio = self.get_strategy()/(self.n*self.n)  # 计算当前合作者比例
        self.cooperation_trend.append(cooperation_ratio)  # 将当前合作者比例加入列表
        # if self.step_count % 1000 == 0:  # 每隔20步绘制一次蒙特卡洛仿真结果
        #     self.plot_simulation()
        # if self.step_count == 10000:  # 如果仿真结束，绘制合作者比例趋势图
        #     print(self.cooperation_trend)
        #     self.plot_cooperation_trend()

    def get_strategy(self):
        #print(cp.sum(self.strategy_matrix))
        return cp.sum(self.strategy_matrix)
    def get_cooperation_ratio(self):
        # print(self.cooperation_trend)
        return self.cooperation_trend

data = []
for j in range(1):
    my_lg = Lattice_Game(100, 100, 3.3)
    for i in range(10000):
        my_lg.step()
    data.append(my_lg.get_cooperation_ratio()[-10:])
sums = [sum(sublist)/10 for sublist in data]
print(sums)
end_time = time.time()
elapsed_time = end_time - start_time  # 计算运行时间
print(f"Simulation took {elapsed_time} seconds")