
from collections import deque
import logging
from typing import List
from app.types.factory_graph import FactoryGraph, SmallStage
from app.types.factory_inventory import FactoryInventoryItem

logger = logging.getLogger(__name__)

def compute_max_outputs(factory_graph: FactoryGraph, inventory: List[FactoryInventoryItem], priorities: dict[int, float]) -> dict[int, float]:
    max_outputs: dict[int, float] = {} # stage_output_id -> quantity
    stage_output_ids = find_stage_output_ids(factory_graph)
    
    for stage_output_id in stage_output_ids:
        inventory_dict = {item.component.id: item.quantity for item in inventory if item.component is not None}
        max_output = compute_max_output_new(factory_graph, inventory_dict, stage_output_ids[stage_output_id])
        desired_output = max_output[stage_output_id] * priorities[stage_output_ids[stage_output_id]]
        max_outputs[stage_output_id] = desired_output

    return max_outputs

def compute_max_output_new(
    factory_graph: FactoryGraph, 
    inventory: dict[int, float],
    stage_id: int
) -> dict[int, float]:
    dependency_subtree = determine_dependency_subtree(factory_graph, stage_id)
    sorted_stages = topological_sort(dependency_subtree)

    max_outputs = {}

    for stage_id in sorted_stages:
        stage = factory_graph.nodes[stage_id].small_stage
        min_input_ratio = float("inf")

        # Determine max possible input ratio
        for stage_input in stage.stage_inputs:
            input_component_id = stage_input.component_id
            required_quantity = stage_input.quantity_per_stage or 0
            available_quantity = inventory.get(input_component_id, 0) or 0
            input_ratio = available_quantity / required_quantity if required_quantity > 0 else float("inf")

            if input_ratio < min_input_ratio:
                min_input_ratio = input_ratio

        min_input_ratio = min(min_input_ratio, 1.0)

        # Allocate based on min input ratio
        for stage_input in stage.stage_inputs:
            input_component_id = stage_input.component_id
            required_quantity = stage_input.quantity_per_stage or 0
            allocation = min_input_ratio * required_quantity
            inventory[input_component_id] -= allocation
            stage_input.allocated_quantity = allocation

        # Determine output quantities
        for stage_output in stage.stage_outputs:
            output_quantity = min_input_ratio * (stage_output.quantity_per_stage or 0)
            max_outputs[stage_output.id] = output_quantity
            output_component_id = stage_output.component_id
            if output_component_id in inventory:
                inventory[output_component_id] += output_quantity
            elif output_component_id is not None:
                inventory[output_component_id] = output_quantity

    return max_outputs


def determine_dependency_subtree(factory_graph: FactoryGraph, stage_id: int) -> FactoryGraph:
    relevant_nodes = set()
    relevant_edges = set()

    def traverse_graph(current_stage_id: int):
        if current_stage_id in relevant_nodes:
            return
        relevant_nodes.add(current_stage_id)

        # Check all edges to see if the current stage is a destination in any edge
        for stage, edges in factory_graph.adj_list.items():
            for edge in edges:
                if edge.outgoing_factory_stage_id == current_stage_id:
                    relevant_edges.add(edge)
                    # Recurse through the graph from the source stage of this edge
                    traverse_graph(edge.incoming_factory_stage_id)

    traverse_graph(stage_id)

    filtered_nodes = {node_id: factory_graph.nodes[node_id] for node_id in relevant_nodes}
    filtered_adj_list = {node_id: [edge for edge in factory_graph.adj_list.get(node_id, []) if edge in relevant_edges] 
                         for node_id in relevant_nodes}
    
    return FactoryGraph(nodes=filtered_nodes, adj_list=filtered_adj_list)


def topological_sort(factory_graph: FactoryGraph) -> List[int]:
    # Step 1: Compute in-degrees
    in_degree = {node_id: 0 for node_id in factory_graph.nodes}
    for node in factory_graph.nodes:
        for edge in factory_graph.adj_list.get(node, []):
            in_degree[edge.outgoing_factory_stage_id] += 1

    # Step 2: Initialize queue
    zero_in_degree_queue = deque([node for node in factory_graph.nodes if in_degree[node] == 0])

    # Step 3: Topological sort
    topological_order = []

    while zero_in_degree_queue:
        node = zero_in_degree_queue.popleft()
        topological_order.append(node)

        # Update in-degrees
        for edge in factory_graph.adj_list.get(node, []):
            neighbor = edge.outgoing_factory_stage_id
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree_queue.append(neighbor)

    # Make sure there is no cycle
    if len(topological_order) != len(factory_graph.nodes):
        raise ValueError("Cycle detected in the graph")
    
    return topological_order


def find_stage_output_ids(factory_graph: FactoryGraph) -> dict[int, int]: # stage_output_id -> stage_id
    stage_output_ids: dict[int, int] = {}
    for node in factory_graph.nodes.values():
        stage = node.small_stage
        for output in stage.stage_outputs:
            stage_output_ids[output.id] = stage.id

    return stage_output_ids


def determine_min_input_ratio(stage: SmallStage, inventory: dict[int, float]) -> float:
    min_input_ratio = float("inf")

    # Determine max possible input ratio
    for stage_input in stage.stage_inputs:
        input_component_id = stage_input.component_id
        required_quantity = stage_input.quantity_per_stage or 0
        available_quantity = inventory.get(input_component_id, 0) or 0
        input_ratio = available_quantity / required_quantity if required_quantity > 0 else float("inf")

        if input_ratio < min_input_ratio:
            min_input_ratio = input_ratio

    return min_input_ratio