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
    file_path = 'data/sfc_datasets.json'
    datasets = load_dataset(file_path)
    processed_datasets = process_datasets(datasets)
    one_dataset = processed_datasets[0]
    context = Context()
    context.physical_network_init(one_dataset['network_resources'], one_dataset['network_topology'])
    context.SFC_init(one_dataset['sfcs'])
    S = ghm_place_sfc(context)
    print(S)

