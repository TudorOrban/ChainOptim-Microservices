

from typing import List
from app.types.factory_graph import FactoryGraph
from app.types.factory_inventory import FactoryInventoryItem
from app.types.production_history_model import ProductionHistory


def process_input_data(factory_graph: FactoryGraph, production_history: ProductionHistory):
    record_size = production_history.daily_production_records.__len__()



def determine_resource_allocations(factory_graph: FactoryGraph, inventory: List[FactoryInventoryItem]):
    pass    