# -*- coding: utf-8 -*-
"""
@Time ： 2024/9/4 00:12
@Auth ： xiaolongtuan
@File ：ilp_placement.py
"""
import numpy as np
import pulp as lp

from dataset_loader import load_dataset, process_datasets, single_bandwidth_resource, single_propagation_delay, \
    single_transmission_delay, min_bandwidth_resource, max_dalay

M = 1000


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
                            # 强制如果f和f_prime分别部署在不同物理节点n_s和n_d上，那么虚拟链路必须通过物理链路，即n_s出度>=1,n_t入度>=1
                            self.model += lp.lpSum(self.Y[i, f, f_prime, n_s, n_p] for n_p in range(self.num_nodes) if
                                                   (n_s, n_p) in self.L) >= 1 - M * (
                                                  2 - self.X[i, f, n_s] - self.X[i, f_prime, n_t])
                            self.model += lp.lpSum(self.Y[i, f, f_prime, n_p, n_t] for n_p in range(self.num_nodes) if
                                                   (n_p, n_t) in self.L) >= 1 - M * (
                                                  2 - self.X[i, f, n_s] - self.X[i, f_prime, n_t])

                            # 中间节点的流量守恒约束：对于中间节点，流入等于流出
                            for n in range(self.num_nodes):
                                if n != n_s and n != n_t:
                                    flow_in = lp.lpSum(self.Y[i, f, f_prime, n_p, n] for n_p in range(self.num_nodes) if
                                                       (n_p, n) in self.L)
                                    flow_out = lp.lpSum(
                                        self.Y[i, f, f_prime, n, n_p] for n_p in range(self.num_nodes) if
                                        (n, n_p) in self.L)
                                    self.model += (flow_in - flow_out <= M * (
                                            2 - self.X[i, f, n_s] - self.X[i, f_prime, n_t]))
                                    self.model += (flow_out - flow_in <= M * (
                                            2 - self.X[i, f, n_s] - self.X[i, f_prime, n_t]))
                                    # lp.lpSum(self.Y[i, f, f_prime, n, n_p] for n_p in range(self.num_nodes) if (n, n_p) in self.L)
                                    # == lp.lpSum(self.Y[i, f, f_prime, n_p, n] for n_p in range(self.num_nodes) if (n_p, n) in self.L)
                                    # if self.X[i, f, n_s]==1， self.X[i, f_prime, n_t] == 1

                            # 虚拟链路的起点和终点必须与 VNF 的物理部署位置一致
                            self.model += (self.Y[i, f, f_prime, n_s, n_t] <= self.X[i, f, n_s])
                            self.model += (self.Y[i, f, f_prime, n_s, n_t] <= self.X[i, f_prime, n_t])
                        else:
                            self.model += (self.Y[i, f, f_prime, n_s, n_t] == 0)  # 不占用链路资源

    def solve(self):
        self.model.solve()
        if lp.LpStatus[self.model.status] == 'Optimal':
            print("Solution is optimal")
            return True
        else:
            print("Solution is not optimal")
            return False

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


def run_solver_on_dataset(dataset):
    # 从数据集中提取网络拓扑和资源信息
    network_topology = dataset['network_topology']
    network_resources = dataset['network_resources']
    sfc_data = dataset['sfcs']
    adjacency_list = dataset['adjacency_list']

    # 提取物理节点和链路的资源
    num_nodes = len(network_topology.nodes)
    node_cpu_capacity = [network_resources[n]['cpu_resource'] for n in range(num_nodes)]
    node_mem_capacity = [network_resources[n]['memory_resource'] for n in range(num_nodes)]
    node_storage_capacity = [network_resources[n]['storage_resource'] for n in range(num_nodes)]

    for n in range(num_nodes):
        for n_prime in range(num_nodes):
            if n == n_prime:  # 同一设备
                network_resources[(n, n_prime)] = {
                    'bandwidth_resource': single_bandwidth_resource,
                    'propagation_delay': single_propagation_delay,
                    'transmission_delay': single_transmission_delay
                }
            if ((n, n_prime) not in network_resources) or ((n_prime, n) not in network_resources):  # 没有链路
                network_resources[(n, n_prime)] = {
                    'bandwidth_resource': min_bandwidth_resource,
                    'propagation_delay': max_dalay,
                    'transmission_delay': max_dalay
                }

    link_capacity = {k: v['bandwidth_resource']
                     for k, v in network_resources.items() if isinstance(k, tuple)}
    link_propagation_delay = {k: v['propagation_delay']
                              for k, v in network_resources.items() if isinstance(k, tuple)}
    link_transmission_delay = {k: v['transmission_delay']
                               for k, v in network_resources.items() if isinstance(k, tuple)}

    # 提取SFC信息
    num_sfc = len(sfc_data)
    vnf_per_sfc = [sfc['num_vnf'] for sfc in sfc_data]
    max_vnf_per_sfc = max(vnf_per_sfc)

    # 创建资源需求矩阵 R_cpu, R_memory, R_storage
    R_cpu = np.zeros((num_sfc, max_vnf_per_sfc))  # CPU资源矩阵
    R_memory = np.zeros((num_sfc, max_vnf_per_sfc))  # 内存资源矩阵
    R_storage = np.zeros((num_sfc, max_vnf_per_sfc))  # 存储资源矩阵
    # 创建虚拟链路带宽需求矩阵 R_link(i, f, f')
    R_link = np.zeros((num_sfc, max_vnf_per_sfc, max_vnf_per_sfc))
    tolerable_delay = np.zeros((num_sfc,))  # 可容忍的延迟
    safety_factor = np.zeros((num_sfc,))

    for i, sfc in enumerate(sfc_data):
        tolerable_delay[i] = sfc['max_latency']
        safety_factor[i] = sfc['safety_factor']

        for f, vnf in enumerate(sfc['vnfs']):
            R_cpu[i, f] = vnf['cpu_consumption']
            R_memory[i, f] = vnf['memory_consumption']
            R_storage[i, f] = vnf['storage_consumption']
        for f, link in enumerate(sfc['links']):
            bandwidth = link['bandwidth_consumption']
            R_link[i, f, f + 1] = bandwidth

    # 创建求解器并求解
    solver = SFCPlacementSolver(sfc_data, num_sfc, max_vnf_per_sfc, num_nodes, node_cpu_capacity, node_mem_capacity,
                                node_storage_capacity, link_capacity, link_propagation_delay,
                                link_transmission_delay,
                                R_cpu, R_memory, R_storage, R_link, tolerable_delay, safety_factor, adjacency_list)
    if solver.solve():
        results = solver.get_results()
        # print(results)
        return results
    else:
        return None


if __name__ == '__main__':
    if __name__ == '__main__':
        # 加载和处理数据集
        file_path = 'data/sfc_datasets_mini.json'
        datasets = load_dataset(file_path)
        processed_datasets = process_datasets(datasets)
        for dataset in processed_datasets:
            results = run_solver_on_dataset(dataset)
            if results is not None:
                for i, f, n in results['placement']:
                    if i == 0:
                        print((i, f, n))
                for i, f, f_prime, n, n_prime in results['links']:
                    if i == 0:
                        print((i, f, f_prime, n, n_prime))
