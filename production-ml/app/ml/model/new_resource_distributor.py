
import logging
from typing import List
from app.ml.model.resource_distributor import determine_dependency_subtree, find_stage_output_ids, topological_sort
from app.types.factory_graph import FactoryGraph
from app.types.factory_inventory import FactoryInventoryItem

logger = logging.getLogger(__name__)

def compute_max_outputs_new(factory_graph: FactoryGraph, inventory: List[FactoryInventoryItem], priorities: dict[int, float]):
    max_outputs: dict[int, float] = {} # stage_output_id -> quantity
    stage_output_ids = find_stage_output_ids(factory_graph)
    
    for stage_output_id in stage_output_ids:
        inventory_dict = {item.component.id: item.quantity for item in inventory if item.component is not None}
        max_output = compute_max_output_new(factory_graph, inventory_dict, priorities, stage_output_id, stage_output_ids[stage_output_id])
        max_outputs.update(max_output)

    return max_outputs

def compute_max_output_new(factory_graph: FactoryGraph, inventory: dict[int, float], priorities: dict[int, float], stage_output_id: int, stage_id: int):
    dependency_subtree = determine_dependency_subtree(factory_graph, stage_id, stage_output_id)
    logger.info(f"Dependency subtree: {dependency_subtree}")
    sorted_stages = topological_sort(dependency_subtree)
    logger.info(f"Sorted stages: {sorted_stages}")

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

        logger.info(f"Stage ID: {stage_id}, min_input_ratio: {min_input_ratio}, stage outputs: {stage.stage_outputs}")

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
            logger.info(f"Output quantity: {output_quantity} for stage output ID: {stage_output.id}")
            max_outputs[stage_output.id] = output_quantity
            output_component_id = stage_output.component_id
            if output_component_id in inventory:
                inventory[output_component_id] += output_quantity
            elif output_component_id is not None:
                inventory[output_component_id] = output_quantity

    logger.info(f"Max outputs: {max_outputs}")
    return max_outputs