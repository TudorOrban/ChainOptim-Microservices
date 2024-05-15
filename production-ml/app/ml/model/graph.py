import dgl
import torch

from app.types.factory_graph import FactoryGraph


def build_graph(factory_graph: FactoryGraph):
    g = dgl.DGLGraph()
    g.add_nodes(len(factory_graph.nodes))

    for src, edges in factory_graph.adj_list.items():
        for edge in edges:
            g.add_edges(src, edge.outgoing_factory_stage_id)

    num_feats = 5
    features = torch.zeros((len(factory_graph.nodes), num_feats))

    for idx, (stage_id, stage_node) in enumerate(factory_graph.nodes.items()):
        features[idx, 0] = stage_node.number_of_steps_capacity or 0
        features[idx, 1] = stage_node.per_duration or 0

    return g, features


def build_heterogeneous_graph(factory_graph: FactoryGraph):
    graph_data = {
        ('stage', 'has_input', 'input'): ([], []), # (src_nodes, dst_nodes)
        ('stage', 'has_output', 'output'): ([], []),
        ('input', 'input_to', 'stage'): ([], []),
        ('output', 'output_from', 'stage'): ([], [])
    }

    node_data = {
        'stage': {'features': [], 'ids': []},
        'input': {'features': [], 'ids': []},
        'output': {'features': [], 'ids': []}
    }

    node_counts = {
        'stage': 0,
        'input': 0,
        'output': 0
    }

    for stage_id, stage_node in factory_graph.nodes.items():
        stage_features = [
            stage_node.number_of_steps_capacity or 0,
            stage_node.per_duration or 0
        ]
        node_data['stage']['features'].append(stage_features)
        node_data['stage']['ids'].append(stage_id)
        node_counts['stage'] += 1

        for input in stage_node.small_stage.stage_inputs:
            input_features = [
                input.component_id or 0,
                input.quantity_per_stage or 0
            ]
            node_data['input']['features'].append(input_features)
            node_data['input']['ids'].append(input.id)
            node_counts['input'] += 1
            
            graph_data[('stage', 'has_input', 'input')][0].append(stage_id)
            graph_data[('stage', 'has_input', 'input')][1].append(input.id)
            graph_data[('input', 'input_to', 'stage')][0].append(input.id)
            graph_data[('input', 'input_to', 'stage')][1].append(stage_id)

        for output in stage_node.small_stage.stage_outputs:
            output_features = [
                output.component_id or 0,
                output.quantity_per_stage or 0
            ]
            node_data['output']['features'].append(output_features)
            node_data['output']['ids'].append(output.id)
            node_counts['output'] += 1

            graph_data[('stage', 'has_output', 'output')][0].append(stage_id)
            graph_data[('stage', 'has_output', 'output')][1].append(output.id)
            graph_data[('output', 'output_from', 'stage')][0].append(output.id)
            graph_data[('output', 'output_from', 'stage')][1].append(stage_id)

    print("Total nodes by type:")
    for ntype in node_data:
        print(f"{ntype}: {len(node_data[ntype]['ids'])} nodes")

    print("Total features by type:")
    for ntype in node_data:
        print(f"{ntype}: {len(node_data[ntype]['features'])} feature sets")

    print("Graph edges detail:")
    for edge_type, edges in graph_data.items():
        print(f"{edge_type}: {len(edges[0])} edges")
        
    num_nodes_dict = {
        'stage': max(node_data['stage']['ids']) + 1,
        'input': max(node_data['input']['ids']) + 1,
        'output': max(node_data['output']['ids']) + 1
    }
    # Inspect graph_data
    print("Graph data structure:")
    for edge_type, edges in graph_data.items():
        print(f"{edge_type}: {len(edges[0])} edges")
        print(f"Source nodes: {edges[0]}")
        print(f"Destination nodes: {edges[1]}")
    
    g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict) # type: ignore

    print(f"Graph node types: {g.ntypes}")

    print(f"num_nodes_dict: {num_nodes_dict}")

    for ntype in node_counts:
        print(f"Assigning features for node type: {ntype}")
        num_nodes = node_counts[ntype]
        feature_list = node_data[ntype]['features']
        print(f"Debug Info - {ntype}: Expected {num_nodes}, got {len(feature_list)} features")
        print(f"Feature List: {feature_list}")
        print(f"IDs in last loop: {node_data[ntype]['ids']}")
        # Validate the node type and inspect ntypes
        print(f"Graph node types: {g.ntypes}")
        if ntype not in g.ntypes:
            print(f"Error: Node type {ntype} not found in graph node types {g.ntypes}")
        else:
            print(f"Node type {ntype} is valid")

        # Inspect feature_list
        print(f"Feature list for {ntype}: {feature_list}")

        print(g)
        
        try:
            g.nodes[ntype].data['features'] = torch.tensor(node_data[ntype]['features'], dtype=torch.float32)
            print("Successfully assigned features to 'output'.")
        except Exception as e:
            print("Failed to assign features to 'output':", e)


    return g