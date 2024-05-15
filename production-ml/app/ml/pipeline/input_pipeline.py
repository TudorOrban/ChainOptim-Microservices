

from typing import Any, Dict, List
from app.types.factory_graph import Edge, FactoryGraph


def apply_unified_id_mapping(factory_graph: FactoryGraph, id_map: Dict[str, Dict[int, int]]) -> FactoryGraph:
    new_nodes = {}
    new_adj_list = {}

    # Remap nodes
    for old_id, node in factory_graph.nodes.items():
        new_id = id_map["stages"][old_id]
        new_nodes[new_id] = node
        node.small_stage.id = new_id  # Assign the new ID to small_stage

        for stage_input in node.small_stage.stage_inputs:
            if stage_input.id in id_map["inputs"]:
                stage_input.id = id_map["inputs"][stage_input.id]
        for stage_output in node.small_stage.stage_outputs:
            if stage_output.id in id_map["outputs"]:
                stage_output.id = id_map["outputs"][stage_output.id]

    # Remap edges
    for old_stage_id, edges in factory_graph.adj_list.items():
        new_stage_id = id_map["stages"][old_stage_id]
        new_adj_list[new_stage_id] = []
        for edge in edges:
            new_edge = Edge(
                incoming_factory_stage_id=id_map["stages"].get(edge.incoming_factory_stage_id, edge.incoming_factory_stage_id),
                incoming_stage_output_id=id_map["outputs"].get(edge.incoming_stage_output_id, edge.incoming_stage_output_id),
                outgoing_factory_stage_id=id_map["stages"].get(edge.outgoing_factory_stage_id, edge.outgoing_factory_stage_id),
                outgoing_stage_input_id=id_map["inputs"].get(edge.outgoing_stage_input_id, edge.outgoing_stage_input_id)
            )
            new_adj_list[new_stage_id].append(new_edge)

    factory_graph.nodes = new_nodes
    factory_graph.adj_list = new_adj_list

    return factory_graph


def unify_ids(factory_graph: FactoryGraph) -> Dict[str, Dict[int, int]]:
    stage_id_counter, input_id_counter, output_id_counter = 0, 0, 0
    id_map = {"stages": {}, "inputs": {}, "outputs": {}}

    for stage_id in factory_graph.nodes.keys():
        id_map["stages"][stage_id] = stage_id_counter
        stage_id_counter += 1

    for stage in factory_graph.nodes.values():
        for stage_input in stage.small_stage.stage_inputs:
            if stage_input.id not in id_map["inputs"]:
                id_map["inputs"][stage_input.id] = input_id_counter
                input_id_counter += 1
        for stage_output in stage.small_stage.stage_outputs:
            if stage_output.id not in id_map["outputs"]:
                id_map["outputs"][stage_output.id] = output_id_counter
                output_id_counter += 1

    return id_map