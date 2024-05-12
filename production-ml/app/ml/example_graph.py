import dgl
import torch
import numpy as np

def build_graph():
    num_nodes = 10
    num_edges = 20
    num_features = 5

    g = dgl.graph((torch.randint(0, num_nodes, (num_edges,)), torch.randint(0, num_nodes, (num_edges,))))

    g.ndata['feat'] = torch.randn(num_nodes, num_features)

    g.edata['weight'] = torch.randn(num_edges, num_features)

    return g

graph = build_graph()