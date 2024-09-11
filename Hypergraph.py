# -*- coding: utf-8 -*-
"""
@Time ： 2024/8/30 00:32
@Auth ： xiaolongtuan
@File ：Hypergraph.py
"""
import itertools
from itertools import combinations
from typing import List, Tuple

import networkx as nx

from context import Context
from entity import VNF, Node


# 我可以为vnf实例定义id，如果是共享vnf，则其实例id相同了，我就讲一个3维的放置转换为了2维，厉害
# 定义超图模型
class Hypergraph:
    def __init__(self):
        self.VNFs = set()  # 顶点集合 V
        self.Nodes = set()

        self.hyperedges = {}  # 超边与sfc id的映射, 排序即为虚拟链路的顺序 {[(f1,n1),(f2,n2)]:sfc_id}
        self.W = {}  # 超边与权重的映射

    def add_VNF(self, v):
        self.VNFs.add(v)

    def add_node(self, node):
        self.Nodes.add(node)

    def add_hyperedge(self, hyper_edge: List[tuple], sfc_id, context: Context):
        for place in hyper_edge:
            self.add_VNF(place[0])
            self.add_node(place[1])

        self.hyperedges[hyper_edge] = sfc_id
        self.W[hyper_edge] = context.sfcs[sfc_id].cul_cost(hyper_edge, context.physical_network)  # 计算超边权重

    def get_weight(self, hyper_edge):
        return self.hyperedges.get(hyper_edge, None)

    def get_edges(self) -> List[Tuple]:
        return self.hyperedges.keys()  # 返回超边列表S

    def initialize_hyperedges(self, context: Context):
        # 遍历所有 VNF 的部署可能性组合
        for id, sfc in context.sfcs.items():
            # 根据约束条件删除不可用的部署可能性
            vnf_placement = []
            for vnf in sfc.virtual_link:
                possible_nodes = []
                for node in context.physical_node_resources:
                    if check_resource_constraints(node, vnf):
                        possible_nodes.append((vnf, node))
                vnf_placement.append(possible_nodes)
            edges = list(itertools.product(*vnf_placement))
            for hyperedge in edges:
                self.add_hyperedge(hyperedge, id, context)
        return

    # 构造冲突图
    def get_conflict_graph(self) -> nx.Graph:
        conflict_graph = nx.Graph()

        # 添加顶点（对应于超边）
        for edge in self.hyperedges:
            conflict_graph.add_node(edge, weight=self.W[edge])

        # 检查超边之间的冲突
        for edge1, edge2 in itertools.combinations(self.hyperedges.keys(), 2):
            # 检查是否共享任何 VNF 或物理节点
            nodes1 = {n for _, n in edge1}
            nodes2 = {n for _, n in edge2}
            vnfs1 = {f for f, _ in edge1}
            vnfs2 = {f for f, _ in edge2}

            # 如果两个超边共享至少一个物理节点或 VNF，则添加冲突边
            if nodes1 & nodes2 or vnfs1 & vnfs2:
                conflict_graph.add_edge(edge1, edge2)

        return conflict_graph


# 构建冲突图
def build_conflict_graph(H: Hypergraph):
    conflict_graph = nx.Graph()

    # 获取所有超边
    edges = H.get_edges()

    # 添加冲突图中的顶点（对应超边）并赋予权重
    for edge in edges:
        conflict_graph.add_node(edge, weight=H.get_weight(edge))

    # 为每对超边创建边，如果它们有冲突（即共享顶点）
    for e1, e2 in combinations(edges, 2):
        if set(e1) & set(e2):  # 如果有交集，说明冲突
            conflict_graph.add_edge(e1, e2)


def check_resource_constraints(physical_node: Node, vnf: VNF):
    return (physical_node.processors >= vnf.processors and
            physical_node.memory >= vnf.memory and
            physical_node.storage >= vnf.storage)
