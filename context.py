# -*- coding: utf-8 -*-
"""
@Time ： 2024/8/31 00:36
@Auth ： xiaolongtuan
@File ：context.py
"""
from dataclasses import dataclass
from typing import List

import networkx as nx
import numpy as np

from entity import Node, SFC, VNF


# from Hypergraph import Hypergraph
# from entity import SFC


@dataclass
class Context:
    sfcs = None  # {id: SFC}
    nfvs = set()  # 所有nfv
    edge_placement_matrix = None
    placement_matrix = None
    physical_network = None
    physical_node_resources = None  # [Node]

    Hypergraph = None

    def physical_network_init(self, network_resources: dict, physical_network: nx.Graph):
        self.physical_node_resources = []
        self.physical_network = physical_network
        for k, value in network_resources.items():
            if isinstance(k, int):
                self.physical_node_resources.append(
                    Node(id=k, processors=value['cpu_resource'], memory=value['memory_resource'],
                         storage=value['storage_resource']))
            elif isinstance(k, tuple):
                self.physical_network[k[0]][k[1]]['latency'] = value['propagation_delay'] + value['transmission_delay']

    def SFC_init(self, sfcs: []):
        self.sfcs = {}
        for id, sfc_data in enumerate(sfcs):
            R_link = {}
            for f, link in enumerate(sfc_data['links']):
                R_link[f, f + 1] = link['bandwidth_consumption']
            sfc = SFC(id=id, name=sfc_data['name'] if 'name' in sfc_data else str(id), bandwidth=R_link,
                      latency=sfc_data['max_latency'], security_level=sfc_data['safety_factor'], virtual_link=[])
            for f, v_data in enumerate(sfc_data['vnfs']):
                v = VNF(id=f, processors=v_data['cpu_consumption'], memory=v_data['memory_consumption'],
                        storage=v_data['storage_consumption'])
                sfc.add_VNF(v)
            self.sfcs[id] = sfc
