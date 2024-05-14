

from collections import deque
from typing import List, Optional
from app.types.factory_graph import Edge, FactoryGraph
from app.types.factory_inventory import FactoryInventoryItem


def compute_max_outputs(factory_graph: FactoryGraph, inventory: List[FactoryInventoryItem], priorities: dict[int, float]):
    max_outputs: dict[int, float] = {} # stage_output_id -> quantity
    stage_output_ids = find_stage_output_ids(factory_graph)
    
    inventory_dict = {item.component.id: item.quantity for item in inventory if item.component is not None}

    for stage_output_id in stage_output_ids:
        compute_max_output(factory_graph, inventory_dict, priorities, stage_output_id, stage_output_ids[stage_output_id])
    
    return max_outputs


def compute_max_output(factory_graph: FactoryGraph, inventory: dict[int, float], priorities: dict[int, float], stage_output_id: int, stage_id: int):
    corresponding_stage = factory_graph.nodes[stage_id].small_stage

    for stage_input in corresponding_stage.stage_inputs:
        component_id = stage_input.component_id
        available_quantity = inventory.get(component_id, 0) or 0

        potential_edge: Optional[Edge] = None

        # Look for connecting stage output
        for node_id, neighbors in factory_graph.adj_list.items():
            for neighbor in neighbors:
                if neighbor.outgoing_factory_stage_id == stage_id and neighbor.outgoing_stage_input_id == stage_input.id:
                    # Found the connecting stage output
                    potential_edge = neighbor

        if potential_edge is None:
            allocation = min(available_quantity, stage_input.quantity_per_stage or 0)
            inventory[component_id] -= allocation
            continue

        # Recursively compute max output for the connecting stage
        compute_max_output(factory_graph, inventory, priorities, potential_edge.incoming_stage_output_id, potential_edge.incoming_factory_stage_id)



def find_stage_output_ids(factory_graph: FactoryGraph) -> dict[int, int]: # stage_output_id -> stage_id
    stage_output_ids: dict[int, int] = {}
    for node in factory_graph.nodes.values():
        stage = node.small_stage
        for output in stage.stage_outputs:
            stage_output_ids[output.id] = stage.id

    return stage_output_ids



def determine_dependency_subtree(factory_graph: FactoryGraph, stage_id: int, stage_output_id: int) -> FactoryGraph:
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



# def topological_sort(factory_graph: FactoryGraph) -> List[int]:
#     # Step 1: Compute in-degrees
#     in_degree = {node_id: 0 for node_id in factory_graph.nodes}
#     for node in factory_graph.nodes:
#         for edge in factory_graph.adj_list.get(node, []):
#             in_degree[edge.outgoing_factory_stage_id] += 1

#     # Step 2: Initialize queue
#     zero_in_degree_queue = deque([node for node in factory_graph.nodes if in_degree[node] == 0])

#     # Step 3: Topological sort
#     topological_order = []

#     while zero_in_degree_queue:
