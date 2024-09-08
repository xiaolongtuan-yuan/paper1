# -*- coding: utf-8 -*-
"""
@Time ： 2024/8/29 23:32
@Auth ： xiaolongtuan
@File ：net_sim.py
"""
from entity import SFC
import networkx as nx
import numpy as np


def calculate_queue_delay_1(utilization, transmission_delay):
    if utilization >= 1:
        return float('inf')  # 避免无穷大的延迟
    return utilization * transmission_delay / (1 - utilization)


def calculate_processing_delay_1(node, vnf_complexity):
    cpu_utilization = (node['initial_processors'] - node['processors']) / node['initial_processors']
    return cpu_utilization * vnf_complexity


def calculate_end_to_end_delay_1(sfc: SFC, physical_network, placement_matrix):
    total_delay = 0
    i = sfc.id
    # 遍历SFC的每一条虚拟链路
    for u, v in sfc.edges():
        source_node = None
        target_node = None

        # 寻找sdc的源节点和目标节点
        for f in range(placement_matrix.shape[1]):
            for n in range(placement_matrix.shape[2]):
                if placement_matrix[i][f][n] and sfc.nodes[u]['function_type'] == f:
                    source_node = n
                if placement_matrix[i][f][n] and sfc.nodes[v]['function_type'] == f:
                    target_node = n

        if source_node is None or target_node is None:
            raise ValueError("找不到源节点或目标节点")

        # 如果相邻VNF部署在相同物理节点
        if source_node == target_node:
            transmission_delay = 0.1  # 假设为0.1 ms
            propagation_delay = 0.1  # 假设为0.1 ms
            queue_delay = 0
        else:
            # 计算传播延迟和传输延迟
            physical_path = nx.shortest_path(physical_network, source=source_node, target=target_node, weight='latency')
            propagation_delay = sum(physical_network[physical_path[i]][physical_path[i + 1]]['latency'] for i in
                                    range(len(physical_path) - 1))
            transmission_delay = sum(
                physical_network[physical_path[i]][physical_path[i + 1]]['latency'] for i in
                range(len(physical_path) - 1))

            # 计算排队延迟
            utilizations = [
                (sfc[u][v]['bandwidth'] / physical_network[physical_path[i]][physical_path[i + 1]]['bandwidth'], i) for
                i in
                range(len(physical_path) - 1)]
            queue_delay = sum(calculate_queue_delay_1(utilization,
                                                      physical_network[physical_path[i]][physical_path[i + 1]][
                                                          'latency']) for utilization, i in utilizations)

        # 计算处理延迟
        vnf_complexity = sfc.nodes[v]['processors']  # 假设VNF复杂度与处理器需求成正比
        processing_delay = calculate_processing_delay_1(physical_network.nodes[target_node], vnf_complexity)

        total_delay += propagation_delay + transmission_delay + queue_delay + processing_delay

    return total_delay


def calculate_end_to_end_delay(all_sfcs, placement_matrix_X, placement_matrix_Y, physical_network,
                               link_propagation_delay, link_transmission_delay):
    total_delays = {}

    for sfc_id, sfc in enumerate(all_sfcs):
        total_delay = 0

        # 遍历SFC的每一条虚拟链路
        for l, link in enumerate(sfc['links']):
            f = l
            f_1 = l + 1
            source_node = None
            target_node = None

            # 查找放置在物理节点上的源VNF和目标VNF
            for n in range(placement_matrix_X.shape[2]):
                if placement_matrix_X[sfc_id, f, n] == 1:
                    source_node = n
                if placement_matrix_X[sfc_id, f_1, n] == 1:
                    target_node = n

            if source_node is None or target_node is None:
                raise ValueError(f"找不到源节点 {f} 或目标节点 {f_1}")

            # 如果相邻的VNF部署在相同的物理节点上
            if source_node == target_node:
                transmission_delay = 0.1  # 假设为 0.1 ms
                propagation_delay = 0.1  # 假设为 0.1 ms
                queue_delay = 0
            else:
                # 查找物理网络中源节点到目标节点的最短路径
                physical_path = nx.shortest_path(physical_network, source=source_node, target=target_node,
                                                 weight='latency')

                # 计算传播延迟和传输延迟
                propagation_delay = sum(link_propagation_delay[physical_path[i], physical_path[i + 1]] for i in
                                        range(len(physical_path) - 1))
                transmission_delay = sum(link_transmission_delay[physical_path[i], physical_path[i + 1]] for i in
                                         range(len(physical_path) - 1))

                # 计算排队延迟，考虑所有SFC的虚拟链路在该物理链路上的资源消耗
                utilizations = []
                for i in range(len(physical_path) - 1):
                    total_bandwidth_usage = sum(
                        placement_matrix_Y[sfc_idx, f1, f2, physical_path[i], physical_path[i + 1]] for sfc_idx in
                        range(placement_matrix_Y.shape[0])
                        for f1 in range(placement_matrix_Y.shape[1])
                        for f2 in range(placement_matrix_Y.shape[1]))
                    link_bandwidth = physical_network[physical_path[i]][physical_path[i + 1]]['bandwidth']
                    utilization = total_bandwidth_usage / link_bandwidth
                    utilizations.append(
                        (utilization, physical_network[physical_path[i]][physical_path[i + 1]]['latency']))

                queue_delay = sum(
                    calculate_queue_delay(utilization, base_latency) for utilization, base_latency in utilizations)

            # 计算处理延迟
            vnf_cpu_consumption = sfc['vnfs'][f]['cpu_consumption']
            processing_delay = calculate_processing_delay(physical_network.nodes[source_node], vnf_cpu_consumption)

            # 累加延迟
            total_delay += propagation_delay + transmission_delay + queue_delay + processing_delay
            if f_1 == sfc['num_vnf'] - 1:  # 最后一个vnf
                total_delay += calculate_processing_delay(physical_network.nodes[target_node],
                                                          sfc['vnfs'][f_1]['cpu_consumption'])

        # 保存当前SFC的总延迟
        total_delays[sfc_id] = total_delay

    return total_delays


def calculate_queue_delay(utilization, base_latency):
    """
    计算排队延迟：利用率越高，排队延迟越大。
    :param utilization: 链路利用率
    :param base_latency: 链路的基础延迟
    :return: 排队延迟
    """
    if utilization >= 1:
        return np.inf  # 链路超载时，排队延迟无穷大
    return base_latency / (1 - utilization)


def calculate_processing_delay(node_resources, vnf_cpu_consumption):
    """
    计算处理延迟：处理延迟与VNF的CPU需求成正比。
    :param node_resources: 物理节点的资源，包含节点的CPU总资源。
    :param vnf_cpu_consumption: 当前VNF的CPU需求。
    :return: 处理延迟
    """
    available_cpu = node_resources['cpu']  # 物理节点的可用CPU资源
    if available_cpu <= 0:
        return np.inf  # 如果没有可用CPU资源，则处理延迟无穷大
    return 1000 * vnf_cpu_consumption / available_cpu  # 假设处理延迟与VNF的CPU需求成正比，且与物理节点可用CPU资源成反比 us
