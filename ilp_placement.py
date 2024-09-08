# -*- coding: utf-8 -*-
"""
@Time ： 2024/9/4 00:12
@Auth ： xiaolongtuan
@File ：ilp_placement.py
"""
import numpy as np
import pulp as lp


def check_data_validity(data):
    for key, value in data.items():
        if np.any(np.isnan(value)) or np.any(np.isinf(value)):
            raise ValueError(f"Data contains NaN or inf values at key: {key}")


class SFCPlacementSolver:
    def __init__(self, SFCs, num_sfc, max_vnf_per_sfc, num_nodes,  # SFC信息
                 node_cpu_capacity, node_mem_capacity, node_storage_capacity,  # 节点资源限制
                 link_capacity, link_propagation_delay, link_transmission_delay,  # 链路信息
                 R_cpu, R_memory, R_storage, R_link, tolerable_delay, safety_factor, adjacency_list):  # 资源需求
        self.SFCs = SFCs
        self.num_sfc = num_sfc
        self.max_vnf_per_sfc = max_vnf_per_sfc
        self.num_nodes = num_nodes
        self.node_cpu_capacity = node_cpu_capacity
        self.node_mem_capacity = node_mem_capacity
        self.node_storage_capacity = node_storage_capacity
        self.link_capacity = link_capacity
        self.link_propagation_delay = link_propagation_delay
        self.link_transmission_delay = link_transmission_delay
        self.R_cpu = R_cpu
        self.R_memory = R_memory
        self.R_storage = R_storage
        self.R_link = R_link
        self.tolerable_delay = tolerable_delay
        self.safety_factor = safety_factor
        self.L = adjacency_list

        # 初始化求解模型
        self.model = lp.LpProblem("SFC_Placement", lp.LpMinimize)

        # 定义变量X(i, f, n)
        self.X = lp.LpVariable.dicts("X",
                                     ((i, f, n) for i in range(self.num_sfc)
                                      for f in range(self.max_vnf_per_sfc)
                                      for n in range(self.num_nodes)),
                                     cat=lp.LpBinary)

        # 定义变量Y(f, f', n, n')
        self.Y = lp.LpVariable.dicts("Y",
                                     ((i, f, f_prime, n, n_prime) for i in range(self.num_sfc)
                                      for f in range(self.max_vnf_per_sfc)
                                      for f_prime in range(self.max_vnf_per_sfc)
                                      for n in range(self.num_nodes)
                                      for n_prime in range(self.num_nodes)),
                                     cat=lp.LpBinary)

        # 添加目标函数
        # 计算节点资源占用
        node_resource_usage = lp.lpSum(
            self.X[(i, f, n)] * (self.R_cpu[i, f] + self.R_memory[i, f] + self.R_storage[i, f])
            for i in range(self.num_sfc)
            for f in range(self.max_vnf_per_sfc)
            for n in range(self.num_nodes))

        # 计算链路资源占用
        link_resource_usage = lp.lpSum(self.Y[(i, f, f_prime, n, n_prime)] * self.R_link[i, f, f_prime]
                                       for i in range(self.num_sfc)
                                       for f in range(self.max_vnf_per_sfc - 1)
                                       for f_prime in range(f + 1, self.max_vnf_per_sfc)
                                       for n in range(self.num_nodes)
                                       for n_prime in range(self.num_nodes))

        # 设置目标函数：最小化资源占用
        self.model += node_resource_usage + link_resource_usage, "Minimize_Resource_Usage"
        self.add_placement_constraints()

    def add_placement_constraints(self):
        # 约束1：每个VNF必须放置在一个物理节点上
        for i in range(self.num_sfc):
            for f in range(self.max_vnf_per_sfc):
                self.model += lp.lpSum(self.X[i, f, n] for n in range(self.num_nodes)) == 1

        # 约束2：每个物理节点上的资源消耗不能超过节点容量
        for n in range(self.num_nodes):
            self.model += lp.lpSum(self.X[i, f, n] * self.R_cpu[i][f] for i in range(self.num_sfc)
                                   for f in range(self.max_vnf_per_sfc)) <= self.node_cpu_capacity[n]
            self.model += lp.lpSum(self.X[i, f, n] * self.R_memory[i][f] for i in range(self.num_sfc)
                                   for f in range(self.max_vnf_per_sfc)) <= self.node_mem_capacity[n]
            self.model += lp.lpSum(self.X[i, f, n] * self.R_storage[i][f] for i in range(self.num_sfc)
                                   for f in range(self.max_vnf_per_sfc)) <= self.node_storage_capacity[n]

        # 约束3：物理链路上的资源消耗不能超过链路带宽， 双向的链路占用之和不能超过链路容量
        for (n, n_prime), capacity in self.link_capacity.items():
            self.model += (lp.lpSum(self.Y[i, f, f_prime, n, n_prime] * self.R_link[i][f][f_prime]
                                    for i in range(self.num_sfc)
                                    for f in range(self.max_vnf_per_sfc)
                                    for f_prime in range(self.max_vnf_per_sfc)) +
                           lp.lpSum(self.Y[i, f, f_prime, n_prime, n] * self.R_link[i][f][f_prime]
                                    for i in range(self.num_sfc)
                                    for f in range(self.max_vnf_per_sfc)
                                    for f_prime in range(self.max_vnf_per_sfc))) <= capacity
        # 约束4：每个SFC的延迟不能超过容忍延迟
        for i, sfc in enumerate(self.SFCs):
            total_delay = 0

            # 遍历 SFC 的虚拟链路
            for f in range(sfc['num_vnf'] - 1):
                f_prime = f + 1

                # 传播和传输延迟
                propagation_delay = lp.lpSum(
                    self.Y[(i, f, f_prime, n, n_prime)] * (
                        self.link_propagation_delay[(n, n_prime)] if (n, n_prime) in self.link_propagation_delay else
                        self.link_propagation_delay[(n_prime, n)])
                    for n in range(self.num_nodes) for n_prime in range(self.num_nodes))

                transmission_delay = lp.lpSum(
                    self.Y[(i, f, f_prime, n, n_prime)] * (
                        self.link_transmission_delay[(n, n_prime)] if (n, n_prime) in self.link_transmission_delay else
                        self.link_transmission_delay[(n_prime, n)])
                    for n in range(self.num_nodes) for n_prime in range(self.num_nodes))

                # 排队延迟
                queue_delay = lp.lpSum(self.Y[(i, f, f_prime, n, n_prime)] * (
                        self.R_link[i, f, f_prime] /
                        (self.link_capacity[(n, n_prime)] if (n, n_prime) in self.link_capacity else self.link_capacity[
                            (n_prime, n)]))
                                       for n in range(self.num_nodes) for n_prime in range(self.num_nodes))

                # 处理延迟
                processing_delay = lp.lpSum(self.X[(i, f, n)] * (self.R_cpu[i, f] / self.node_cpu_capacity[n])
                                            for n in range(self.num_nodes))

                total_delay += propagation_delay + transmission_delay + queue_delay + processing_delay

            # 最后一个VNF的处理延迟
            total_delay += lp.lpSum(
                self.X[(i, sfc['num_vnf'] - 1, n)] * (self.R_cpu[i, sfc['num_vnf'] - 1] / self.node_cpu_capacity[n])
                for n in range(self.num_nodes))

            # 添加约束：确保端到端延迟小于或等于 SFC 的可容忍延迟
            self.model += total_delay <= sfc['max_latency'], f"Delay_Constraint_SFC_{i}"

        # 约束5 链路部署
        # 为每条虚拟链路 ff' 添加路径连续性和流量守恒约束
        for i in range(self.num_sfc):  # 遍历所有 SFC
            for f in range(self.max_vnf_per_sfc - 1):  # f 是第一个 VNF
                f_prime = f + 1  # f' 是相邻的 VNF
                for n_s in range(self.num_nodes):  # n_s 是 f 的物理部署节点
                    for n_t in range(self.num_nodes):  # n_t 是 f' 的物理部署节点
                        if n_s != n_t:
                            # 源节点流出约束：从源节点 n_s 必须有一条物理链路流出
                            self.model += (
                                    lp.lpSum(self.Y[i, f, f_prime, n_s, n_p] for n_p in range(self.num_nodes) if
                                             (n_s, n_p) in self.L) == 1
                            )
                            # 目的节点流入约束：必须有一条物理链路流入目的节点 n_t
                            self.model += (
                                    lp.lpSum(self.Y[i, f, f_prime, n_p, n_t] for n_p in range(self.num_nodes) if
                                             (n_p, n_t) in self.L) == 1
                            )
                            # 中间节点的流量守恒约束：对于中间节点，流入等于流出
                            for n in range(self.num_nodes):
                                if n != n_s and n != n_t:
                                    self.model += (
                                            lp.lpSum(self.Y[i, f, f_prime, n_p, n] for n_p in range(self.num_nodes) if
                                                     (n_p, n) in self.L) ==
                                            lp.lpSum(self.Y[i, f, f_prime, n, n_p] for n_p in range(self.num_nodes) if
                                                     (n, n_p) in self.L)
                                    )
                            # 虚拟链路的起点和终点必须与 VNF 的物理部署位置一致
                            self.model += (self.Y[i, f, f_prime, n_s, n_t] <= self.X[i, f, n_s])
                            self.model += (self.Y[i, f, f_prime, n_s, n_t] <= self.X[i, f_prime, n_t])
                        else:
                            self.model += (self.Y[i, f, f_prime, n_s, n_t] == 0)

    def solve(self):
        self.model.solve()
        if lp.LpStatus[self.model.status] == 'Optimal':
            print("Solution is optimal")
        else:
            print("Solution is not optimal")

    def get_results(self):
        # 提取结果
        placement_results = []
        link_results = []

        for i in range(self.num_sfc):
            for f in range(self.max_vnf_per_sfc):
                for n in range(self.num_nodes):
                    if lp.value(self.X[i, f, n]) == 1:
                        placement_results.append((i, f, n))

        for i in range(self.num_sfc):
            for f in range(self.max_vnf_per_sfc):
                for f_prime in range(self.max_vnf_per_sfc):
                    for n in range(self.num_nodes):
                        for n_prime in range(self.num_nodes):
                            if lp.value(self.Y[i, f, f_prime, n, n_prime]) == 1:
                                link_results.append((i, f, f_prime, n, n_prime))

        return {'placement': placement_results, 'links': link_results}


if __name__ == '__main__':
    # 假设的输入数据
    num_sfc = 2  # SFC数量
    vnf_per_sfc = [3, 2]  # 每个SFC中的VNF数量
    num_nodes = 4  # 物理网络中的节点数量
    num_links = 5  # 物理网络中的链路数量

    # 资源限制（例如节点资源和链路带宽）
    node_capacity = [10, 15, 20, 10]  # 每个物理节点的资源
    link_capacity = {(0, 1): 10, (1, 2): 10, (2, 3): 15, (0, 2): 10, (1, 3): 20}  # 每条物理链路的带宽

    # VNF的资源需求（每个SFC的每个VNF在每个物理节点上的需求, 假设为常量1）
    vnf_demand = 1

    # 建立优化模型
    model = lp.LpProblem("SFC_Placement", lp.LpMinimize)

    # 定义变量X(i, f, n)
    X = lp.LpVariable.dicts("X",
                            ((i, f, n) for i in range(num_sfc)
                             for f in range(vnf_per_sfc[i])
                             for n in range(num_nodes)),
                            cat=lp.LpBinary)

    # 定义变量Y(f, f', n, n')
    Y = lp.LpVariable.dicts("Y",
                            ((i, f, f_prime, n, n_prime) for i in range(num_sfc)
                             for f in range(vnf_per_sfc[i])
                             for f_prime in range(vnf_per_sfc[i])
                             for n in range(num_nodes)
                             for n_prime in range(num_nodes)),
                            cat=lp.LpBinary)

    # 目标函数：最小化资源使用 (这里只是示例，可以根据具体问题定义目标函数)
    model += lp.lpSum(X)

    # 约束条件

    # 1. 每个VNF只能放置在一个物理节点上
    for i in range(num_sfc):
        for f in range(vnf_per_sfc[i]):
            model += lp.lpSum(X[i, f, n] for n in range(num_nodes)) == 1

    # 2. 每个物理节点上的VNF资源消耗不能超过节点容量
    for n in range(num_nodes):
        model += lp.lpSum(X[i, f, n] * vnf_demand for i in range(num_sfc)
                          for f in range(vnf_per_sfc[i])) <= node_capacity[n]  # 这个嵌套循环返回关于变量n的列表，对其求和表示该节点上的所有VNF的资源消耗

    # 3. 每个物理链路上的虚拟链路资源消耗不能超过链路带宽
    for (n, n_prime) in link_capacity:
        model += lp.lpSum(Y[i, f, f_prime, n, n_prime] for i in range(num_sfc)
                          for f in range(vnf_per_sfc[i])
                          for f_prime in range(vnf_per_sfc[i])) <= link_capacity[(n, n_prime)]

    # 4. VNF之间的虚拟链路只能放置在物理链路上，如果这两个VNF已放置在相应的物理节点上
    for i in range(num_sfc):
        for f in range(vnf_per_sfc[i] - 1):
            for n in range(num_nodes):
                for n_prime in range(num_nodes):
                    model += Y[i, f, f + 1, n, n_prime] <= X[i, f, n]
                    model += Y[i, f, f + 1, n, n_prime] <= X[i, f + 1, n_prime]

                    model += Y[i, f, f, n, n_prime] == 0

    # 求解模型
    model.solve()

    # 输出结果
    for i in range(num_sfc):
        for f in range(vnf_per_sfc[i]):
            for n in range(num_nodes):
                if lp.value(X[i, f, n]) == 1:
                    print(f"SFC {i}, VNF {f} is placed on node {n}")
    for i in range(num_sfc):
        for f in range(vnf_per_sfc[i]):
            for f_prime in range(vnf_per_sfc[i]):
                for n in range(num_nodes):
                    for n_prime in range(num_nodes):
                        if lp.value(Y[i, f, f_prime, n, n_prime]) == 1:
                            print(f"Virtual link {f}-{f_prime} is placed on physical link {n}-{n_prime}")
