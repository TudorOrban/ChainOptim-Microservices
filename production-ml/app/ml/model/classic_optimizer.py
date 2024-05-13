

from typing import Dict, List
from app.ml.data.data_generator import find_component_ids
from app.types.factory_graph import FactoryGraph, SmallStageInput
from app.types.factory_inventory import FactoryInventoryItem


def determine_optimal_distribution(factory_graph: FactoryGraph, inventory: List[FactoryInventoryItem], priorities: dict[int, float]):
    optimal_distribution: dict[int, float] = {} # stage_input_id -> quantity
    
    componentIds: List[int] = find_component_ids(factory_graph)

    for componentId in componentIds:
        # Find available quantity
        corresponding_inventory_item = next((item for item in inventory if item.component is not None and item.component.id == componentId), None)
        if corresponding_inventory_item is None:
            continue
        available_quantity = corresponding_inventory_item.quantity

        # Find total needed quantity
        corresponding_stage_inputs = find_stage_inputs_for_component(factory_graph, componentId)
        total_needed_quantity = sum(input.requested_quantity for input in corresponding_stage_inputs.values() if input.requested_quantity is not None)

        if total_needed_quantity <= available_quantity: # Allocate all
            for input in corresponding_stage_inputs.values():
                optimal_distribution[input.id] = input.requested_quantity or 0 
            break

        current_total_needed_quantity = total_needed_quantity

        # Allocate based on priority
        for stage_id, input in corresponding_stage_inputs.items():
            if input.requested_quantity is None:
                continue
            allocated_amount = input.requested_quantity * priorities[stage_id]
            optimal_distribution[input.id] = allocated_amount
            current_total_needed_quantity -= allocated_amount
            available_quantity -= allocated_amount

        # Continue allocating based on priority
        while current_total_needed_quantity > 0 and available_quantity > 0:
            for stage_id, input in corresponding_stage_inputs.items():
                if input.requested_quantity is None:
                    continue
                remaining_needed_quantity = input.requested_quantity - optimal_distribution[input.id]
                quantity_to_allocate = remaining_needed_quantity * priorities[stage_id]
                if remaining_needed_quantity <= 0 or available_quantity <= quantity_to_allocate:
                    continue

                optimal_distribution[input.id] += quantity_to_allocate
                current_total_needed_quantity -= quantity_to_allocate
                available_quantity -= quantity_to_allocate
                
    return optimal_distribution

def allocate_resource(
    current_total_needed_quantity: float, 
    available_quantity: float, 
    priorities: dict[int, float], 
    corresponding_stage_inputs: dict[int, SmallStageInput], 
    optimal_distribution: dict[int, float]
):
    for stage_id, input in corresponding_stage_inputs.items():
        if input.requested_quantity is None:
            continue
        remaining_needed_quantity = input.requested_quantity - optimal_distribution[input.id]
        quantity_to_allocate = remaining_needed_quantity * priorities[stage_id]
        if remaining_needed_quantity <= 0 or available_quantity <= quantity_to_allocate:
            continue

        optimal_distribution[input.id] += quantity_to_allocate
        current_total_needed_quantity -= quantity_to_allocate
        available_quantity -= quantity_to_allocate


def find_stage_inputs_for_component(factory_graph: FactoryGraph, componentId: int) -> dict[int, SmallStageInput]:
    stage_inputs: Dict[int, SmallStageInput] = {}
    
    for node in factory_graph.nodes.values():
        for input in node.small_stage.stage_inputs:
            if input.component_id == componentId and input.requested_quantity is not None:
                stage_inputs[node.small_stage.id] = input

    return stage_inputs