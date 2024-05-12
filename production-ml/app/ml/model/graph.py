import dgl

from app.types.factory_graph import FactoryGraph


def build_graph(factory_graph: FactoryGraph):
    g = dgl.DGLGraph()
    
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(factory_graph.nodes.keys())}
    g.add_nodes(len(factory_graph.nodes))

    for src_id, edges in factory_graph.adj_list.items():
        src_idx = node_id_to_idx[src_id]
        for edge in edges:
            dst_idx = node_id_to_idx[edge.outgoing_factory_stage_id]
            g.add_edge(src_idx, dst_idx)

    return g
