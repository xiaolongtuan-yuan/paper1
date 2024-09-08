# -*- coding: utf-8 -*-
"""
@Time ： 2024/8/30 20:30
@Auth ： xiaolongtuan
@File ：entity.py
"""
from dataclasses import dataclass
from typing import List

import networkx as nx

alpha = 1
beta = 1
gamma = 1
delta = 1


@dataclass(frozen=True)
class VNF:
    id: int
    processors: int
    memory: int
    storage: int


@dataclass(frozen=True)
class Node:
    id: int
    processors: int
    memory: int
    storage: int


@dataclass
class SFC:
    id: int
    name: str
    bandwidth: dict
    latency: int
    security_level: int
    virtual_link: list

    def add_VNF(self, v: VNF):
        self.virtual_link.append(v)

    def cul_cost(self, placement_map: List[tuple], physical_network: nx.Graph):
        # 根据物理链路、链路放置方案计算超边的成本
        cost = 0
        cost += placement_map[0][0].processors * alpha
        cost += placement_map[0][0].memory * beta
        cost += placement_map[0][0].storage * gamma

        for i in range(0, len(placement_map) - 1):
            n1 = placement_map[i][1]
            n2 = placement_map[i + 1][1]
            cost += placement_map[i + 1][0].processors * alpha
            cost += placement_map[i + 1][0].memory * beta
            cost += placement_map[i + 1][0].storage * gamma

            if n1.id != n2.id:
                cost += nx.shortest_path_length(physical_network, source=n1.id, target=n2.id,
                                                weight='latency') * self.bandwidth[
                            (placement_map[i][0].id, placement_map[i + 1][0].id)] * delta
        return cost
