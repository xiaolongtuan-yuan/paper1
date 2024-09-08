# -*- coding: utf-8 -*-
"""
@Time ： 2024/9/6 21:59
@Auth ： xiaolongtuan
@File ：dataset_loader.py
"""

import json
import networkx as nx
import numpy as np

from ilp_placement import SFCPlacementSolver

# 同一设备下的传输性能
single_bandwidth_resource = 100 * 1024
single_propagation_delay = 0
single_transmission_delay = 0

max_dalay = 1e6  # us
min_bandwidth_resource = 1


def load_dataset(file_path):
    with open(file_path, 'r') as f:
        datasets = json.load(f)
    return datasets


def convert_edges_to_tuples(network_resources):
    converted_resources = {}
    L = set()  # 临接列表
    for key, value in network_resources.items():
        if '-' in key:  # Edge keys are in the form of "u-v"
            u, v = map(int, key.split('-'))
            converted_resources[(u, v)] = value
            L.add((u, v))
            L.add((v, u))
        else:  # Node keys
            converted_resources[int(key)] = value
    return converted_resources, L


def reconstruct_network_topology(network_topology_data):
    G = nx.node_link_graph(network_topology_data)
    # Convert node IDs to integers if they are not already
    G = nx.relabel_nodes(G, {node: int(node) for node in G.nodes})
    return G


def process_datasets(datasets):
    processed_datasets = []
    for dataset in datasets:
        # Convert edge keys from strings to tuples
        network_resources, L = convert_edges_to_tuples(dataset['network_resources'])

        # Reconstruct the network topology from node-link data
        network_topology = reconstruct_network_topology(dataset['network_topology'])

        # Append processed dataset
        processed_datasets.append({
            'network_topology': network_topology,
            'network_resources': network_resources,
            'sfcs': dataset['sfcs'],
            'adjacency_list': L
        })
    return processed_datasets


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
    solver.solve()
    results = solver.get_results()
    print(results)

    return results


if __name__ == '__main__':
    # 加载和处理数据集
    file_path = 'data/sfc_datasets.json'
    datasets = load_dataset(file_path)
    processed_datasets = process_datasets(datasets)
    run_solver_on_dataset(processed_datasets[0])
    # 示例：打印处理后的第一个数据集的内容
    import pprint

    # pprint.pprint(processed_datasets[0])
