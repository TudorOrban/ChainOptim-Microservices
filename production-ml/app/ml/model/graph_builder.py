import dgl
import torch

from app.types.factory_graph import FactoryGraph, StageNode

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
        
    num_nodes_dict = {
        'stage': max(node_data['stage']['ids']) + 1,
        'input': max(node_data['input']['ids']) + 1,
        'output': max(node_data['output']['ids']) + 1
    }
    # Create a DGL graph
    g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict) # type: ignore

    for ntype in node_data:
        try:
            if node_data[ntype]['features']:
                g.nodes[ntype].data['features'] = torch.tensor(node_data[ntype]['features'], dtype=torch.float32)
                print(f"Successfully assigned features to '{ntype}'.")
            else:
                print(f"No features to assign for '{ntype}'.")
        except Exception as e:
            print(f"Failed to assign features to '{ntype}': {e}")

    return g
