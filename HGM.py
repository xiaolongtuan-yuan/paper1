# -*- coding: utf-8 -*-
"""
@Time ： 2024/9/7 22:44
@Auth ： xiaolongtuan
@File ：HGM.py
"""
import itertools

import networkx as nx
import numpy as np
import pulp as lp

from dataset_loader import load_dataset, process_datasets, single_bandwidth_resource, single_propagation_delay, \
    single_transmission_delay, min_bandwidth_resource, max_dalay


class HyperGraph:
    def __init__(self):
        self.nodes = set()  # 超图中的节点集
        self.hyperedges = []  # 超边集

    def add_node(self, node):
        self.nodes.add(node)

    def add_hyperedge(self, nodes, weight):
        self.hyperedges.append((nodes, weight))


class HGMSolver:
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

    def create_hypergraph_with_xy(self):
        H = HyperGraph()

        # Step 1: 初始化 X 和 Y 变量
        X = {}  # X[(i, f, n)] 表示第 i 个 SFC 的第 f 个 VNF 是否部署在物理节点 n 上
        Y = {}  # Y[(i, f, f', n, n')] 表示第 i 个 SFC 的虚拟链路 f -> f' 是否经过物理链路 n -> n'

        # Step 2: 遍历所有 SFC 请求，定义 X 和 Y 变量
        for i, sfc in enumerate(self.SFCs):
            # 初始化 X 变量：遍历每个 VNF 和物理节点
            for f, vnf in enumerate(sfc['vnfs']):
                for n in range(self.num_nodes):
                    X[(i, f, n)] = lp.LpVariable(f"X_{i}_{f}_{n}", 0, 1, lp.LpBinary)

            # 初始化 Y 变量：遍历每个虚拟链路和物理链路
            for f in range(len(sfc['vnfs'])):
                for f_prime in range(len(sfc['vnfs'])):
                    for n in range(self.num_nodes):  # n_s 是 f 的物理部署节点
                        for n_prime in range(self.num_nodes):  # n_t 是 f' 的物理部署节点
                            Y[(i, f, f_prime, n, n_prime)] = lp.LpVariable(f"Y_{i}_{f}_{f_prime}_{n}_{n_prime}", 0, 1,
                                                                           lp.LpBinary)

        # Step 3: 计算超边的权重
        for i, sfc in enumerate(self.SFCs):
            mapped_nodes = []
            total_weight = 0
            # 计算节点资源消耗
            for f, vnf in enumerate(sfc['vnfs']):
                for n in range(self.num_nodes):
                    total_weight += X[(i, f, n)] * (self.R_cpu[i, f] + self.R_memory[i, f] + self.R_storage[i, f])

            # 计算链路资源消耗
            for f in range(len(sfc['vnfs']) - 1):
                f_prime = f + 1
                for n in range(self.num_nodes):
                    for n_prime in range(self.num_nodes):
                        total_weight += Y[(i, f, f_prime, n, n_prime)] * self.R_link[i, f, f_prime]

            for f, vnf in enumerate(sfc['vnfs']):
                for n in range(self.num_nodes):
                    if self.node_cpu_capacity[n] >= vnf['cpu_consumption'] and \
                            self.node_mem_capacity[n] >= vnf['memory_consumption'] and \
                            self.node_storage_capacity[n] >= vnf['storage_consumption']:
                        H.add_node((i, f, n))
                        mapped_nodes.append((i, f, n))

            # 为每个 SFC 添加超边，权重为总资源消耗
            H.add_hyperedge(mapped_nodes, total_weight)
        return H, X, Y

    # 算法1 寻找权重最大的独立集合
    def ilp_solve(self):
        # 创建优化问题
        problem = lp.LpProblem("Minimize_Hyperedge_Weights", lp.LpMinimize)

        # 从 create_hypergraph_with_xy 方法获取超图、X 和 Y 变量
        hypergraph, X, Y = self.create_hypergraph_with_xy()

        # Step 2: 目标函数 - 最小化超边权重
        total_weight = lp.lpSum([weight for i, (nodes, weight) in enumerate(hypergraph.hyperedges)])

        problem += total_weight  # 设置目标函数

        # Step 3: 添加约束
        problem = self.add_placement_constraints(problem, X, Y)

        # 求解优化问题
        problem.solve()

        # 输出结果
        print(f"Status: {lp.LpStatus[problem.status]}")
        # for var in problem.variables():
        #     print(f"{var.name} = {var.varValue}")
        self.get_results(X, Y)

        return problem

    def add_placement_constraints(self, problem, X, Y):
        # 约束1：每个VNF必须放置在一个物理节点上
        for i, sfc in enumerate(self.SFCs):
            for f, vnf in enumerate(sfc['vnfs']):
                problem += lp.lpSum(X[(i, f, n)] for n in range(self.num_nodes)) == 1

        # 约束2：每个物理节点上的资源消耗不能超过节点容量
        for n in range(self.num_nodes):
            problem += lp.lpSum(X[(i, f, n)] * self.R_cpu[i][f] for i, sfc in enumerate(self.SFCs) for f, _ in
                                enumerate(sfc['vnfs'])) <= self.node_cpu_capacity[n]
            problem += lp.lpSum(X[(i, f, n)] * self.R_memory[i][f] for i, sfc in enumerate(self.SFCs) for f, _ in
                                enumerate(sfc['vnfs'])) <= self.node_mem_capacity[n]
            problem += lp.lpSum(X[(i, f, n)] * self.R_storage[i][f] for i, sfc in enumerate(self.SFCs) for f, _ in
                                enumerate(sfc['vnfs'])) <= self.node_storage_capacity[n]

        # 约束3：物理链路上的资源消耗不能超过链路带宽， 双向的链路占用之和不能超过链路容量
        for (n, n_prime), capacity in self.link_capacity.items():
            problem += (lp.lpSum(Y[(i, f, f_prime, n, n_prime)] * self.R_link[i][f][f_prime]
                                 for i, sfc in enumerate(self.SFCs)
                                 for f, _ in enumerate(sfc['vnfs'])
                                 for f_prime, _ in enumerate(sfc['vnfs'])) +
                        lp.lpSum(Y[(i, f, f_prime, n_prime, n)] * self.R_link[i][f][f_prime]
                                 for i, sfc in enumerate(self.SFCs)
                                 for f, _ in enumerate(sfc['vnfs'])
                                 for f_prime, _ in enumerate(sfc['vnfs']))) <= capacity
        # 约束4：每个SFC的延迟不能超过容忍延迟
        for i, sfc in enumerate(self.SFCs):
            total_delay = 0

            # 遍历 SFC 的虚拟链路
            for f in range(sfc['num_vnf'] - 1):
                f_prime = f + 1

                # 传播和传输延迟
                propagation_delay = lp.lpSum(
                    Y[(i, f, f_prime, n, n_prime)] * (
                        self.link_propagation_delay[(n, n_prime)] if (n, n_prime) in self.link_propagation_delay else
                        self.link_propagation_delay[(n_prime, n)])
                    for n in range(self.num_nodes) for n_prime in range(self.num_nodes))

                transmission_delay = lp.lpSum(
                    Y[(i, f, f_prime, n, n_prime)] * (
                        self.link_transmission_delay[(n, n_prime)] if (n, n_prime) in self.link_transmission_delay else
                        self.link_transmission_delay[(n_prime, n)])
                    for n in range(self.num_nodes) for n_prime in range(self.num_nodes))

                # 排队延迟
                queue_delay = lp.lpSum(Y[(i, f, f_prime, n, n_prime)] * (
                        self.R_link[i, f, f_prime] /
                        (self.link_capacity[(n, n_prime)] if (n, n_prime) in self.link_capacity else self.link_capacity[
                            (n_prime, n)]))
                                       for n in range(self.num_nodes) for n_prime in range(self.num_nodes))

                # 处理延迟
                processing_delay = lp.lpSum(X[(i, f, n)] * (self.R_cpu[i, f] / self.node_cpu_capacity[n])
                                            for n in range(self.num_nodes))

                total_delay += propagation_delay + transmission_delay + queue_delay + processing_delay

            # 最后一个VNF的处理延迟
            total_delay += lp.lpSum(
                X[(i, sfc['num_vnf'] - 1, n)] * (self.R_cpu[i, sfc['num_vnf'] - 1] / self.node_cpu_capacity[n])
                for n in range(self.num_nodes))

            # 添加约束：确保端到端延迟小于或等于 SFC 的可容忍延迟
            problem += total_delay <= sfc['max_latency'], f"Delay_Constraint_SFC_{i}"

        # 约束5 链路部署
        # 为每条虚拟链路 ff' 添加路径连续性和流量守恒约束
        for i, sfc in enumerate(self.SFCs):
            for f, _ in enumerate(sfc['vnfs'][:-1]):
                f_prime = f + 1  # f' 是相邻的 VNF
                for n_s in range(self.num_nodes):  # n_s 是 f 的物理部署节点
                    for n_t in range(self.num_nodes):  # n_t 是 f' 的物理部署节点
                        if n_s != n_t:
                            '''
                            # 源节点流出约束：从源节点 n_s 必须有一条物理链路流出
                            problem += (
                                    lp.lpSum(Y[i, f, f_prime, n_s, n_p] for n_p in range(self.num_nodes) if
                                             (n_s, n_p) in self.L) == 1
                            )
                            # 目的节点流入约束：必须有一条物理链路流入目的节点 n_t
                            problem += (
                                    lp.lpSum(Y[i, f, f_prime, n_p, n_t] for n_p in range(self.num_nodes) if
                                             (n_p, n_t) in self.L) == 1
                            )
                            # 中间节点的流量守恒约束：对于中间节点，流入等于流出
                            for n in range(self.num_nodes):
                                if n != n_s and n != n_t:
                                    problem += (
                                            lp.lpSum(Y[i, f, f_prime, n_p, n] for n_p in range(self.num_nodes) if
                                                     (n_p, n) in self.L) ==
                                            lp.lpSum(Y[i, f, f_prime, n, n_p] for n_p in range(self.num_nodes) if
                                                     (n, n_p) in self.L)
                                    )
                            '''
                            # 虚拟链路的起点和终点必须与 VNF 的物理部署位置一致
                            problem += (Y[i, f, f_prime, n_s, n_t] <= X[i, f, n_s])
                            problem += (Y[i, f, f_prime, n_s, n_t] <= X[i, f_prime, n_t])
                        else:
                            problem += (Y[i, f, f_prime, n_s, n_t] == 0)
        return problem

    def HGM_solve(self):
        H, X, Y = self.create_hypergraph_with_xy()
        conflict_graph = self.create_conflict_graph(H)
        S = self.vertex_set_search_algorithm(conflict_graph)
        S = self.phi_claw_local_search_algorithm(S, conflict_graph)

        return S

    def get_results(self, X, Y):
        # 提取结果
        placement_results = []
        link_results = []

        for i, sfc in enumerate(self.SFCs):
            for f, _ in enumerate(sfc['vnfs']):
                for n in range(self.num_nodes):
                    if lp.value(X[(i, f, n)]) == 1:
                        placement_results.append((i, f, n))

        for i, sfc in enumerate(self.SFCs):
            for f, _ in enumerate(sfc['vnfs']):
                for f_prime, _ in enumerate(sfc['vnfs']):
                    for n in range(self.num_nodes):
                        for n_prime in range(self.num_nodes):
                            if lp.value(Y[(i, f, f_prime, n, n_prime)]) == 1:
                                link_results.append((i, f, f_prime, n, n_prime))
        print({'placement': placement_results, 'links': link_results})
        return

    def create_conflict_graph(self, hypergraph: HyperGraph):
        """
        根据给定的超图 H 构造冲突图 φ。

        :param hypergraph: 已构建的超图 H，其中每个超边有与资源占用相关的权重。
        :return: 冲突图 φ
        """
        # 初始化冲突图 φ
        conflict_graph = nx.Graph()

        # Step 1: 添加超图中的每个超边作为冲突图中的顶点，并设置权重
        hyperedges = hypergraph.hyperedges
        for i, (nodes, weight) in enumerate(hyperedges):
            conflict_graph.add_node(i, weight=weight)

        # Step 2: 检查超图中哪些超边共享节点，构建冲突边
        for edge1, edge2 in itertools.combinations(self.hyperedges, 2):
            # 检查是否共享任何 VNF 或物理节点
            nodes1 = {n for _, _, n in edge1}
            nodes2 = {n for _, _, n in edge2}
            # 如果两个超边共享至少一个物理节点或 VNF，则添加冲突边
            if nodes1 & nodes2:
                conflict_graph.add_edge(edge1, edge2)
        return conflict_graph

    def vertex_set_search_algorithm(self, conflict_graph):
        # Initialization: Initialize the independent set S
        S = set()
        Z = set(conflict_graph.nodes)

        while Z:
            # Choose the vertex with the maximum weight from Z
            z_max = max(Z, key=lambda z: conflict_graph.nodes[z]['weight'])
            S.add(z_max)

            # Find all adjacent vertices of z_max
            B_max = set(conflict_graph.neighbors(z_max))

            # Update Z by removing z_max and its neighbors
            Z -= {z_max}
            Z -= B_max

        # Return the selected independent set S
        return S

    # 算法2
    def phi_claw_local_search_algorithm(self, S, conflict_graph):
        # Step 1: Rank vertices in S based on their weights
        S = sorted(S, key=lambda v: conflict_graph.nodes[v]['weight'], reverse=True)

        i = 0
        while i < len(S):
            current_vertex = S[i]
            B_i = list(conflict_graph.neighbors(current_vertex))
            B_i = sorted(B_i, key=lambda v: conflict_graph.nodes[v]['weight'], reverse=True)

            # Step 5: Set the i-th vertex as the center vertex of 𝜙-claw
            for phi in range(1, 4):
                found_claw = False

                for j in range(len(B_i)):
                    candidate_set = set([current_vertex] + B_i[:phi])

                    # Check if candidate set forms a valid 𝜙-claw in the conflict graph
                    if all(not conflict_graph.has_edge(x, y) for x in candidate_set for y in candidate_set if x != y):
                        found_claw = True
                        new_S = (set(S) - set(B_i)) | candidate_set

                        # Check if the weight of the new set is greater than the current set
                        if sum(conflict_graph.nodes[v]['weight'] for v in new_S) > sum(
                                conflict_graph.nodes[v]['weight'] for v in S):
                            S = new_S
                            i = 0  # Restart the search from the beginning with the new S
                            break

                if found_claw:
                    break

            i += 1

        return S


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
    hgm_solver = HGMSolver(sfc_data, num_sfc, max_vnf_per_sfc, num_nodes, node_cpu_capacity, node_mem_capacity,
                           node_storage_capacity, link_capacity, link_propagation_delay,
                           link_transmission_delay,
                           R_cpu, R_memory, R_storage, R_link, tolerable_delay, safety_factor, adjacency_list)
    hgm_solver.ilp_solve()


if __name__ == '__main__':
    file_path = 'data/sfc_datasets.json'
    datasets = load_dataset(file_path)
    processed_datasets = process_datasets(datasets)
    run_solver_on_dataset(processed_datasets[0])
