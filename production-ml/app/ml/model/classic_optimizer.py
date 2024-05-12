

from typing import List
from app.types.factory_graph import FactoryGraph
from app.types.factory_inventory import FactoryInventoryItem


def determine_optimal_distribution(factory_graph: FactoryGraph, inventory: List[FactoryInventoryItem], priorities: dict[int, float]):
    optimal_distribution: dict[int, float] = {} # stage_input_id -> quantity
    