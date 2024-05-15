

from collections import deque
from typing import List, Optional
from app.ml.model.resource_distributor import find_stage_output_ids
from app.types.factory_graph import Edge, FactoryGraph
from app.types.factory_inventory import FactoryInventoryItem


def compute_max_outputs_old(factory_graph: FactoryGraph, inventory: List[FactoryInventoryItem], priorities: dict[int, float]):
    max_outputs: dict[int, float] = {} # stage_output_id -> quantity
    stage_output_ids = find_stage_output_ids(factory_graph)
    
    inventory_dict = {item.component.id: item.quantity for item in inventory if item.component is not None}

    for stage_output_id in stage_output_ids:
        compute_max_output_old(factory_graph, inventory_dict, priorities, stage_output_id, stage_output_ids[stage_output_id])
    
    return max_outputs


def compute_max_output_old(factory_graph: FactoryGraph, inventory: dict[int, float], priorities: dict[int, float], stage_output_id: int, stage_id: int):
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
        compute_max_output_old(factory_graph, inventory, priorities, potential_edge.incoming_stage_output_id, potential_edge.incoming_factory_stage_id)


