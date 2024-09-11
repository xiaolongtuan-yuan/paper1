# -*- coding: utf-8 -*-
"""
@Time ： 2024/8/29 23:35
@Auth ： xiaolongtuan
@File ：placement.py
"""
from typing import List

import networkx as nx
import numpy as np

from Hypergraph import Hypergraph
from context import Context
from entity import SFC


def simple_place_sfc(sfcs: List[SFC], physical_network: nx.DiGraph):
    # 为每个VNF寻找合适的物理节点
    N = len(physical_network.nodes())
    I = len(sfcs)
    F = max([len(sfc.nodes()) for sfc in sfcs])
    # 初始化大小为I*F*N的3维布尔矩阵，初始化值为False
    placement_matrix = [[[False for _ in range(N)] for _ in range(F)] for _ in range(I)]
    # 将placement_matrix转为numpy数组
    placement_matrix = np.array(placement_matrix)

    for i, sfc in enumerate(sfcs):
        for f, vnf in enumerate(sfc.nodes()):
            vnf_requirements = sfc.nodes[vnf]
            # 寻找满足要求的物理节点
            for n, p_node in enumerate(physical_network.nodes()):
                p_node_resources = physical_network.nodes[p_node]

                if (p_node_resources['processors'] >= vnf_requirements['processors'] and
                        p_node_resources['memory'] >= vnf_requirements['memory'] and
                        p_node_resources['storage'] >= vnf_requirements['storage']):
                    # 进行资源分配
                    placement_matrix[i][f][n] = True
                    # 更新物理节点资源
                    physical_network.nodes[p_node]['processors'] -= vnf_requirements['processors']
                    physical_network.nodes[p_node]['memory'] -= vnf_requirements['memory']
                    physical_network.nodes[p_node]['storage'] -= vnf_requirements['storage']
                    break

        # 验证虚拟链路的放置
        for u, v in sfc.edges():
            sfc_bandwidth = sfc[u][v]['bandwidth']
            physical_path = nx.shortest_path(physical_network, source=np.where(placement_matrix[i][u] == True),
                                             target=np.where(placement_matrix[i][v] == True), weight='latency')

            # 检查路径上的带宽是否满足
            for i in range(len(physical_path) - 1):
                if physical_network[physical_path[i]][physical_path[i + 1]]['bandwidth'] < sfc_bandwidth:
                    return False, None

    return True, placement_matrix


def ghm_place_sfc(context: Context):
    # 构建超图，每个超边为i:[(f,n)]，表示sfc i的vnf在png上的一种部署方式
    hypergraph = Hypergraph()
    hypergraph.initialize_hyperedges(context)

    conflict_graph = hypergraph.get_conflict_graph()
    S = vertex_set_search_algorithm(conflict_graph)
    # 寻找最优独立集合
    S = phi_claw_local_search_algorithm(S, conflict_graph)
    # todo 将最大权重子集转换为布置方案 x(f,n)
    return S


# 算法1 寻找权重最大的独立集合
def vertex_set_search_algorithm(conflict_graph):
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
def phi_claw_local_search_algorithm(S, conflict_graph):
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


def hga_place_sfc(sfc, physical_network):
    ...
    # 构建物理网权重图

    # 依次处理到达的服务链
    # 计算物理网的边权重

    # 贪心的寻找权重最小路径

    # 计算路径上节点的权重

    # 依次寻找满足约束且权重最小的物理节点部署vnf


if __name__ == '__main__':
    from dataset_loader import load_dataset, process_datasets

    # 加载和处理数据集
    file_path = 'data/sfc_datasets_mini.json'
    datasets = load_dataset(file_path)
    processed_datasets = process_datasets(datasets)
    one_dataset = processed_datasets[0]
    context = Context()
    context.physical_network_init(one_dataset['network_resources'], one_dataset['network_topology'])
    context.SFC_init(one_dataset['sfcs'])
    S = ghm_place_sfc(context)
    print(S)

