# -*- coding: utf-8 -*-
"""
@Time ： 2024/9/7 22:19
@Auth ： xiaolongtuan
@File ：gpt_test.py
"""
import networkx as nx
import random
import numpy as np


# 创建物理网络拓扑
def create_physical_network(num_nodes):
    G = nx.Graph()

    for i in range(num_nodes):
        # 添加节点，假设每个节点有随机的资源容量
        G.add_node(i, capacity=random.randint(10, 50))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # 每条边都有带宽和延迟
            G.add_edge(i, j, bandwidth=random.randint(10, 100), delay=random.uniform(1, 10))

    return G

# SFC 请求类
class SFCRequest:
    def __init__(self, functions, demand):
        self.functions = functions  # 服务功能链，表示功能节点
        self.demand = demand        # 请求的流量需求

# 创建SFC请求集合
def create_sfc_requests(num_requests, max_functions_per_request):
    requests = []
    for _ in range(num_requests):
        num_functions = random.randint(2, max_functions_per_request)
        functions = [f"F_{i}" for i in range(num_functions)]  # 功能链
        demand = random.randint(1, 10)  # 流量需求
        requests.append(SFCRequest(functions, demand))
    return requests

# 计算超边权重 (延迟)
def calculate_edge_weight(G, path):
    weight = 0
    for i in range(len(path) - 1):
        weight += G[path[i]][path[i+1]]['delay']
    return weight


# 贪心算法实现
def greedy_sfc_placement(G, sfc_requests):
    mapping = {}  # 存储每个功能节点的物理映射

    for sfc in sfc_requests:
        for function in sfc.functions:
            # 查找可用的物理节点，满足资源和带宽约束
            candidate_nodes = [v for v in G.nodes if G.nodes[v]['capacity'] >= sfc.demand]

            # 如果没有符合要求的节点，则失败
            if not candidate_nodes:
                print(f"No available resources for function {function}")
                return None

            # 贪心选择延迟最小的节点
            best_node = None
            min_delay = float('inf')

            for v in candidate_nodes:
                # 假设功能节点直接映射到物理节点，不经过中间链路
                delay = 0
                if len(mapping) > 0:
                    # 计算到已有映射节点的路径延迟
                    for mapped_function, mapped_node in mapping.items():
                        if nx.has_path(G, mapped_node, v):
                            path = nx.shortest_path(G, mapped_node, v, weight='delay')
                            delay += calculate_edge_weight(G, path)

                # 选择延迟最小的节点
                if delay < min_delay:
                    best_node = v
                    min_delay = delay

            # 更新物理节点的容量
            if best_node is not None:
                G.nodes[best_node]['capacity'] -= sfc.demand
                mapping[function] = best_node
            else:
                print(f"Failed to map function {function}")
                return None

    return mapping