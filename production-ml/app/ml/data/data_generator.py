
import random
from typing import List
from datetime import datetime

from app.types.factory_graph import FactoryGraph
from app.types.factory_inventory import Component, FactoryInventoryItem, UnitOfMeasurement

def generate_data(factory_graph: FactoryGraph) -> tuple[List[FactoryInventoryItem], dict[int, float]]:
    component_ids = find_component_ids(factory_graph)
    
    inventory = generate_inventory_items(component_ids)

    stage_ids = list(factory_graph.nodes.keys())
    priorities = generate_priorities(stage_ids)

    return inventory, priorities

def find_component_ids(factory_graph: FactoryGraph) -> List[int]:
    component_ids: List[int] = []

    for node in factory_graph.nodes.values():
        stage = node.small_stage

        for input in stage.stage_inputs:
            component_ids.append(input.component_id)

        for output in stage.stage_outputs:
            if output.component_id not in component_ids and output.component_id != None:
                component_ids.append(output.component_id)

    return component_ids

def generate_inventory_items(component_ids: List[int]) -> List[FactoryInventoryItem]:
    inventory: List[FactoryInventoryItem] = []

    for component_id in component_ids:
        component = Component(
            id=component_id,
            name=f"Component {component_id}",
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
            quantity=max(0, random.gauss(50, 5)),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        inventory.append(item)
    
    return inventory

def generate_priorities(stage_ids: List[int]) -> dict[int, float]:
    priorities: dict[int, float] = {}
    for stage_id in stage_ids:
        priorities[stage_id] = random.random()
    
    priorities_sum = sum(priorities.values())
    if priorities_sum == 0:
        return priorities
    
    for stage_id in stage_ids:
        priorities[stage_id] = priorities[stage_id] / priorities_sum # Normalize

    return priorities
