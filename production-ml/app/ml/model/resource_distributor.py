
import logging
from typing import List
from app.ml.services.graph_service import determine_dependency_subtree, topological_sort
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

def find_stage_output_ids(factory_graph: FactoryGraph) -> dict[int, int]: # stage_output_id -> stage_id
    stage_output_ids: dict[int, int] = {}
    for node in factory_graph.nodes.values():
        stage = node.small_stage
        for output in stage.stage_outputs:
            stage_output_ids[output.id] = stage.id

    return stage_output_ids

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
        min_input_ratio = determine_min_input_ratio(stage, inventory)
        allocate_resource(stage, inventory, min_input_ratio)
        determine_max_output_quantity(stage, inventory, min_input_ratio, max_outputs)

    return max_outputs

def determine_min_input_ratio(stage: SmallStage, inventory: dict[int, float]) -> float:
    min_input_ratio = float("inf")

    for stage_input in stage.stage_inputs:
        input_component_id = stage_input.component_id
        required_quantity = stage_input.quantity_per_stage or 0
        available_quantity = inventory.get(input_component_id, 0) or 0
        input_ratio = available_quantity / required_quantity if required_quantity > 0 else float("inf")

        if input_ratio < min_input_ratio:
            min_input_ratio = input_ratio

    min_input_ratio = min(min_input_ratio, 1.0)

    return min_input_ratio

def allocate_resource(stage, inventory, min_input_ratio):
    for stage_input in stage.stage_inputs:
        input_component_id = stage_input.component_id
        required_quantity = stage_input.quantity_per_stage or 0
        allocation = min_input_ratio * required_quantity
        inventory[input_component_id] -= allocation
        stage_input.allocated_quantity = allocation

def determine_max_output_quantity(stage: SmallStage, inventory: dict[int, float], min_input_ratio: float, max_outputs: dict[int, float]):
    for stage_output in stage.stage_outputs:
        output_quantity = min_input_ratio * (stage_output.quantity_per_stage or 0)
        max_outputs[stage_output.id] = output_quantity
        
        output_component_id = stage_output.component_id
        if output_component_id in inventory:
            inventory[output_component_id] += output_quantity
        elif output_component_id is not None:
            inventory[output_component_id] = output_quantity
