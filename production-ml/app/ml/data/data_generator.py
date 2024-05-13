
import random
from typing import List
from datetime import datetime

from app.types.factory_graph import FactoryGraph
from app.types.factory_inventory import Component, FactoryInventoryItem, UnitOfMeasurement

def gen_data(factory_graph: FactoryGraph):
    componentIds = find_component_ids(factory_graph)
    inventory = generate_inventory_items(componentIds)

    stageIds = list(factory_graph.nodes.keys())
    priorities = generate_priorities(stageIds)

    return inventory, priorities


def find_component_ids(factory_graph: FactoryGraph) -> List[int]:
    componentIds: List[int] = []
    for node in factory_graph.nodes.values():
        stage = node.small_stage
        for input in stage.stage_inputs:
            componentIds.append(input.component_id)
        for output in stage.stage_outputs:
            if output.component_id not in componentIds and output.component_id != None:
                componentIds.append(output.component_id)
    return componentIds

def generate_inventory_items(componentIds: List[int]):
    inventory: List[FactoryInventoryItem] = []

    for componentId in componentIds:
        component = Component(
            id=componentId,
            name=f"Component {componentId}",
            unit=UnitOfMeasurement(
                id=1,
                name="kg"
            )
        )
        item = FactoryInventoryItem(
            id=1,
            factory_id=1,
            component=component,
            product=None,
            quantity=1000,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        inventory.append(item)
    
    return inventory

def generate_priorities(stageIds: List[int]) -> dict[int, float]:
    priorities: dict[int, float] = {}
    for stageId in stageIds:
        priorities[stageId] = random.random()
    
    priorities_sum = sum(priorities.values())
    if priorities_sum == 0:
        return priorities
    for stageId in stageIds:
        priorities[stageId] = priorities[stageId] / priorities_sum # Normalize

    return priorities
