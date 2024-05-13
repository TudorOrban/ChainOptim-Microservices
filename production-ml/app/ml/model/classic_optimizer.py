

from typing import List
from app.ml.data.data_generator import find_component_ids
from app.types.factory_graph import FactoryGraph, SmallStageInput
from app.types.factory_inventory import FactoryInventoryItem


def determine_optimal_distribution(factory_graph: FactoryGraph, inventory: List[FactoryInventoryItem], priorities: dict[int, float]):
    optimal_distribution: dict[int, float] = {} # stage_input_id -> quantity
    
    componentIds = find_component_ids(factory_graph)

    for componentId in componentIds:
        # Find available quantity
        corresponding_inventory_item = next((item for item in inventory if item.component.id == componentId), None)
        if corresponding_inventory_item is None:
            continue
        available_quantity = corresponding_inventory_item.quantity

        # Find total needed quantity
        corresponding_stage_inputs = find_stage_inputs_for_component(factory_graph, componentId)
        total_needed_quantity = sum([input.requested_quantity for input in corresponding_stage_inputs])

        if total_needed_quantity <= available_quantity: # Allocate all
            for input in corresponding_stage_inputs:
                optimal_distribution[input.id] = input.requested_quantity
            break

        current_total_needed_quantity = total_needed_quantity

        # Allocate based on priority
        for input in corresponding_stage_inputs:
            allocated_amount = input.requested_quantity * priorities[input.stage_id]
            optimal_distribution[input.id] = allocated_amount
            current_total_needed_quantity -= allocated_amount
            available_quantity -= allocated_amount

        # Continue allocating based on priority
        while current_total_needed_quantity > 0 & available_quantity > 0:
            for input in corresponding_stage_inputs:
                remaining_needed_quantity = input.requested_quantity - optimal_distribution[input.id]
                quantity_to_allocate = remaining_needed_quantity * priorities[input.stage_id]
                if remaining_needed_quantity <= 0 | available_quantity <= quantity_to_allocate:
                    continue

                optimal_distribution[input.id] += quantity_to_allocate
                current_total_needed_quantity -= quantity_to_allocate
                available_quantity -= quantity_to_allocate
                
    return optimal_distribution

def allocate_resource(
    current_total_needed_quantity: float, 
    available_quantity: float, 
    priorities: dict[int, float], 
    corresponding_stage_inputs: List[SmallStageInput], 
    optimal_distribution: dict[int, float]
):
    for input in corresponding_stage_inputs:
        remaining_needed_quantity = input.requested_quantity - optimal_distribution[input.id]
        quantity_to_allocate = remaining_needed_quantity * priorities[input.stage_id]
        if remaining_needed_quantity <= 0 | available_quantity <= quantity_to_allocate:
            continue

        optimal_distribution[input.id] += quantity_to_allocate
        current_total_needed_quantity -= quantity_to_allocate
        available_quantity -= quantity_to_allocate


def find_stage_inputs_for_component(factory_graph: FactoryGraph, componentId: int):
    stage_inputs = []
    for node in factory_graph.nodes.values():
        stage = node.small_stage
        for input in stage.stage_inputs:
            if input.component_id == componentId:
                stage_inputs.append(input)
    return stage_inputs