# -*- coding: utf-8 -*-
"""
@Time ： 2024/8/29 22:26
@Auth ： xiaolongtuan
@File ：main.py
"""
import networkx as nx

from net_sim import calculate_end_to_end_delay
from placement import simple_place_sfc

# 创建物理网络图
physical_network = nx.Graph()

# 添加节点，节点属性包括处理器、内存、存储资源
physical_network.add_node(1, processors=8, memory=16, storage=500)
physical_network.add_node(2, processors=4, memory=8, storage=250)
physical_network.add_node(3, processors=16, memory=32, storage=1000)

# 添加链路，链路属性包括带宽和传播延迟
physical_network.add_edge(1, 2, bandwidth=10, latency=2)
physical_network.add_edge(2, 3, bandwidth=20, latency=1)
physical_network.add_edge(1, 3, bandwidth=15, latency=3)

# 创建SFC图
sfc_1 = nx.DiGraph()

# 添加VNF节点，节点属性包括处理器、内存、存储需求
sfc_1.add_node(1, processors=2, memory=4, storage=100)
sfc_1.add_node(2, processors=4, memory=8, storage=200)
sfc_1.add_node(3, processors=1, memory=2, storage=50)

# 添加虚拟链路，链路属性包括带宽需求
sfc_1.add_edge(1, 2, bandwidth=5)
sfc_1.add_edge(2, 3, bandwidth=10)

sfcs = [sfc_1]

# 初始化物理节点的初始资源（用于计算利用率）
for node in physical_network.nodes():
    physical_network.nodes[node]['initial_processors'] = physical_network.nodes[node]['processors']

# 放置SFC并计算延迟
success, placement_matrix = simple_place_sfc(sfcs, physical_network)
if success:
    for i, sfc in enumerate(sfcs):
        mapping = {v: k for k, v in enumerate(sfc.nodes())}
    end_to_end_delay = calculate_end_to_end_delay(sfc, physical_network, mapping)
    print(f"SFC successfully placed with mapping: {placement_matrix}")
    print(f"End-to-end delay for the SFC: {end_to_end_delay} ms")
else:
    print("Failed to place SFC")