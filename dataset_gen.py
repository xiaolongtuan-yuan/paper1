# -*- coding: utf-8 -*-
"""
@Time ： 2024/9/4 01:06
@Auth ： xiaolongtuan
@File ：dataset_gen.py
"""
'''
参数设置
Number of SFCs = (5, 25]
Number of VNF pre SFC request = [2, 3]
Number of VNF Types = (1, 10]
CPU consumption of VNF f = (10, 50]MIPS
Memory consumption of VNF f = (10, 50]MB
Storage consumption of VNF f = (10, 20]MB
The bandwidth consumption of link (ff') = (1, 10] Mbps
Available CPU resource for physical node n = (900, 1100] MIPS
Available memory for physical node n = (900, 1100] MB
Available storage for physical node n = (9,11] GB
Available bandwidth resource of physical link (nn') = (9, 11] Gbps
Processing delay of each packet on node n = [0.8, 1] ms
Propagation delay of physical link (nn') = [2, 5] us
Transmission delay of physical link (nn') = (9, 11] us
The maximum tolerable latency of service chain i = [15, 30] ms
Minimum safety factor of service chain i = [0.6, 1.0]
'''
import random
import networkx as nx
import json

# 设置数据集大小
num_datasets = 100

# 数据范围
sfc_range = (3, 6)
node_range = (20, 30)
vnf_per_sfc_range = [4, 6]
vnf_types_range = (1, 10)
cpu_consumption_range = (30, 50)  # MIPS
memory_consumption_range = (30, 50)  # MB
storage_consumption_range = (30, 50)  # MB
bandwidth_consumption_range = (1, 10)  # Mbps
cpu_resource_range = (80, 100)  # MIPS
memory_resource_range = (80, 100)  # MB
storage_resource_range = (80, 100)  # MB (converted from GB)
bandwidth_resource_range = (500, 1000)  # Mbps (converted from Gbps)
propagation_delay_range = (2, 5)  # us
transmission_delay_range = (9, 11)  # us
latency_range = (15000, 30000)  # us
safety_factor_range = (0.6, 1.0)

dataset_name= 'sfc_datasets_miniresource'

def generate_random_graph(num_nodes, num_edges):
    # 生成一个随机的复杂拓扑网络图
    G = nx.erdos_renyi_graph(n=num_nodes, p=num_edges / (num_nodes * (num_nodes - 1) // 2))
    # 确保图是连通的
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n=num_nodes, p=num_edges / (num_nodes * (num_nodes - 1) // 2))
    return G


def generate_network_resources(G):
    resources = {}
    for node in G.nodes:
        resources[int(node)] = {
            'cpu_resource': random.randint(cpu_resource_range[0], cpu_resource_range[1]),
            'memory_resource': random.randint(memory_resource_range[0], memory_resource_range[1]),
            'storage_resource': random.randint(storage_resource_range[0], storage_resource_range[1])
        }
    for u, v in G.edges:
        resources[f"{u}-{v}"] = {
            'bandwidth_resource': random.randint(bandwidth_resource_range[0], bandwidth_resource_range[1]),
            'propagation_delay': random.uniform(propagation_delay_range[0], propagation_delay_range[1]),
            'transmission_delay': random.uniform(transmission_delay_range[0], transmission_delay_range[1])
        }
    return resources


def generate_sfc_data(num_sfc):
    sfcs = []
    for _ in range(num_sfc):
        sfc = {}
        num_vnf = random.choice(vnf_per_sfc_range)
        sfc['num_vnf'] = num_vnf
        vnf_types = random.randint(vnf_types_range[0], vnf_types_range[1])
        sfc['vnf_types'] = vnf_types
        vnfs = []
        for _ in range(num_vnf):
            vnf = {
                'cpu_consumption': random.randint(cpu_consumption_range[0], cpu_consumption_range[1]),
                'memory_consumption': random.randint(memory_consumption_range[0], memory_consumption_range[1]),
                'storage_consumption': random.randint(storage_consumption_range[0], storage_consumption_range[1])
            }
            vnfs.append(vnf)
        sfc['vnfs'] = vnfs
        links = []
        for _ in range(num_vnf - 1):
            link = {
                'bandwidth_consumption': random.randint(bandwidth_consumption_range[0], bandwidth_consumption_range[1])
            }
            links.append(link)
        sfc['links'] = links
        # sfc['processing_delay'] = random.uniform(processing_delay_range[0], processing_delay_range[1])
        sfc['propagation_delay'] = random.uniform(propagation_delay_range[0], propagation_delay_range[1])
        sfc['transmission_delay'] = random.uniform(transmission_delay_range[0], transmission_delay_range[1])
        sfc['max_latency'] = random.randint(latency_range[0], latency_range[1])
        sfc['safety_factor'] = random.uniform(safety_factor_range[0], safety_factor_range[1])
        sfcs.append(sfc)
    return sfcs


# 生成数据集
datasets = []

for _ in range(num_datasets):
    num_nodes = random.randint(node_range[0], node_range[1])  # 随机选择节点数量
    num_edges = random.randint(num_nodes, num_nodes * (num_nodes - 1) // 2)  # 随机选择边数量
    G = generate_random_graph(num_nodes, num_edges)
    network_resources = generate_network_resources(G)

    num_sfc = random.randint(sfc_range[0], sfc_range[1])
    sfcs = generate_sfc_data(num_sfc)

    dataset = {
        'network_topology': nx.node_link_data(G),
        'network_resources': network_resources,
        'sfcs': sfcs
    }

    datasets.append(dataset)

# 保存为JSON文件
with open(f'data/{dataset_name}.json', 'w') as f:
    json.dump(datasets, f, indent=4)
