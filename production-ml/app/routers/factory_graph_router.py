

import logging
from fastapi import APIRouter

from app.services.factory_graph_service import create_factory_graph, get_factory_graph
from app.types.factory_graph import CamelCaseFactoryProductionGraph, FactoryProductionGraph
from app.utils.utils import deep_convert_model_to_snake

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/factory-graphs/")
async def create_factory_graph_endpoint(factory_graph: CamelCaseFactoryProductionGraph):
    logger.info(f"Received factory graph: {factory_graph}")
    factory_graph_snake_case_dict = deep_convert_model_to_snake(factory_graph)
    logger.info(f"Converted to snake case: {factory_graph_snake_case_dict}")
    factory_graph_snake_case = FactoryProductionGraph(**factory_graph_snake_case_dict)
    return create_factory_graph(factory_graph_snake_case)


@router.get("/factory-graphs/{id}")
async def read_factory_graph(id: str):
    return get_factory_graph(id)