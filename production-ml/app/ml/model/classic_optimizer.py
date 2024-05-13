

from typing import Dict, List
from app.ml.data.data_generator import find_component_ids
from app.types.factory_graph import FactoryGraph, SmallStageInput
from app.types.factory_inventory import FactoryInventoryItem


def determine_optimal_distribution(
    factory_graph: FactoryGraph, 
    inventory: List[FactoryInventoryItem], 
    priorities: dict[int, float]
) -> dict[int, float]:
    optimal_distribution: dict[int, float] = {} # stage_input_id -> quantity
    
    
    component_ids: List[int] = find_component_ids(factory_graph)

    for component_id in component_ids:
        # Find available quantity
        corresponding_inventory_item = next((item for item in inventory if item.component is not None and item.component.id == component_id), None)
        if corresponding_inventory_item is None:
            continue
        available_quantity = corresponding_inventory_item.quantity

        # Find total needed quantity
        corresponding_stage_inputs = find_stage_inputs_for_component(factory_graph, component_id)
        total_needed_quantity = sum(input.requested_quantity for input in corresponding_stage_inputs.values() if input.requested_quantity is not None)

        if total_needed_quantity <= available_quantity: # Allocate all
            for input in corresponding_stage_inputs.values():
                optimal_distribution[input.id] = input.requested_quantity or 0 
            continue

        current_total_needed_quantity = total_needed_quantity

        allocate_resource(current_total_needed_quantity, available_quantity, priorities, corresponding_stage_inputs, optimal_distribution, 0)

    return optimal_distribution

def allocate_resource(
    current_total_needed_quantity: float, 
    available_quantity: float, 
    priorities: dict[int, float], 
    corresponding_stage_inputs: dict[int, SmallStageInput], 
    optimal_distribution: dict[int, float],
    operation_count: int = 0
):
    if operation_count > 10:
        return
    
    for stage_id, input in corresponding_stage_inputs.items():
        if available_quantity <= 0 or current_total_needed_quantity <= 0:
            return
        if input.requested_quantity is None:
            continue
        remaining_needed_quantity = input.requested_quantity - optimal_distribution[input.id]
        quantity_to_allocate = remaining_needed_quantity * priorities[stage_id]
        if remaining_needed_quantity <= 0 or available_quantity <= quantity_to_allocate:
            continue

        optimal_distribution[input.id] += quantity_to_allocate
        current_total_needed_quantity -= quantity_to_allocate
        available_quantity -= quantity_to_allocate
    

    operation_count += 1

    if current_total_needed_quantity > 0 and available_quantity > 0:
        allocate_resource(current_total_needed_quantity, available_quantity, priorities, corresponding_stage_inputs, optimal_distribution)




def find_stage_inputs_for_component(factory_graph: FactoryGraph, component_id: int) -> dict[int, SmallStageInput]:
    stage_inputs: Dict[int, SmallStageInput] = {}
    
    for node in factory_graph.nodes.values():
        for input in node.small_stage.stage_inputs:
            if input.component_id == component_id and input.requested_quantity is not None:
                stage_inputs[node.small_stage.id] = input

    return stage_inputs