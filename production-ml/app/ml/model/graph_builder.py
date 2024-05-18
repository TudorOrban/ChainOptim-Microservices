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

    # Track the maximum ID encountered for each node type
    max_ids = {'stage': -1, 'input': -1, 'output': -1}

    for stage_id, stage_node in factory_graph.nodes.items():
        max_ids['stage'] = max(max_ids['stage'], stage_id)
        node_data['stage']['features'].append([
            stage_node.number_of_steps_capacity or 0,
            stage_node.per_duration or 0
        ])
        node_data['stage']['ids'].append(stage_id)

        for input in stage_node.small_stage.stage_inputs:
            input_id = input.id
            max_ids['input'] = max(max_ids['input'], input_id)
            node_data['input']['features'].append([
                input.component_id or 0,
                input.quantity_per_stage or 0
            ])
            node_data['input']['ids'].append(input_id)

            graph_data[('stage', 'has_input', 'input')][0].append(stage_id)
            graph_data[('stage', 'has_input', 'input')][1].append(input_id)
            graph_data[('input', 'input_to', 'stage')][0].append(input_id)
            graph_data[('input', 'input_to', 'stage')][1].append(stage_id)
            
        for output in stage_node.small_stage.stage_outputs:
            output_id = output.id
            max_ids['output'] = max(max_ids['output'], output_id)
            node_data['output']['features'].append([
                output.component_id or 0,
                output.quantity_per_stage or 0
            ])
            node_data['output']['ids'].append(output_id)

            graph_data[('stage', 'has_output', 'output')][0].append(stage_id)
            graph_data[('stage', 'has_output', 'output')][1].append(output_id)
            graph_data[('output', 'output_from', 'stage')][0].append(output_id)
            graph_data[('output', 'output_from', 'stage')][1].append(stage_id)
        
    dummy_ids = {'stage': max_ids['stage'] + 1, 'input': max_ids['input'] + 1, 'output': max_ids['output'] + 1}
    
    # Add dummy nodes with their respective features
    for ntype in ['stage', 'input', 'output']:
        dummy_feature = [0] * len(node_data[ntype]['features'][0])
        node_data[ntype]['features'].append(dummy_feature)
        node_data[ntype]['ids'].append(dummy_ids[ntype])

    # Connect nodes to the appropriate dummy nodes
    for ntype in ['stage', 'input', 'output']:
        for node_id in node_data[ntype]['ids']:
            if ntype == 'stage':
                dummy_input_id = dummy_ids['input']
                dummy_output_id = dummy_ids['output']
                graph_data[('stage', 'has_input', 'input')][0].append(node_id)
                graph_data[('stage', 'has_input', 'input')][1].append(dummy_input_id)
                graph_data[('stage', 'has_output', 'output')][0].append(node_id)
                graph_data[('stage', 'has_output', 'output')][1].append(dummy_output_id)
            elif ntype == 'input':
                dummy_stage_id = dummy_ids['stage']
                graph_data[('input', 'input_to', 'stage')][0].append(node_id)
                graph_data[('input', 'input_to', 'stage')][1].append(dummy_stage_id)
            elif ntype == 'output':
                dummy_stage_id = dummy_ids['stage']
                graph_data[('output', 'output_from', 'stage')][0].append(node_id)
                graph_data[('output', 'output_from', 'stage')][1].append(dummy_stage_id)

    print("max_ids:", max_ids)
    num_nodes_dict = {k: v + 1 for k, v in dummy_ids.items()}  # Correct calculation to +1
    print("num_nodes_dict:", num_nodes_dict)

    for etype, (srcs, dsts) in graph_data.items():
        print(f"Edge type: {etype}")
        print(f"Src nodes: {srcs}")
        print(f"Dst nodes: {dsts}")

    for ntype in node_data:
        print(f"Node type: {ntype}")
        print(f"Node IDs: {node_data[ntype]['ids']}")
        print(f"Node features: {node_data[ntype]['features']}")


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
