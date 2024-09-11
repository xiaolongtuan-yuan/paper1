# -*- coding: utf-8 -*-
"""
@Time ： 2024/9/6 21:59
@Auth ： xiaolongtuan
@File ：dataset_loader.py
"""

import json
import networkx as nx
import numpy as np

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
            L.add((int(key), int(key)))
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