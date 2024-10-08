

import logging
from fastapi import APIRouter

from app.ml.data.data_generator import generate_data
from app.ml.model.resource_distributor import compute_max_outputs, determine_dependency_subtree, topological_sort
from app.services.factory_graph_service import create_factory_graph, get_factory_graph_by_id
from app.types.factory_graph import CamelCaseFactoryProductionGraph, FactoryProductionGraph
from app.utils.utils import deep_convert_model_to_snake

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/api/v1/ml/factory-graphs/")
async def create_factory_graph_endpoint(factory_graph: CamelCaseFactoryProductionGraph):
    logger.info(f"Received factory graph: {factory_graph}")
    factory_graph_snake_case_dict = deep_convert_model_to_snake(factory_graph)
    logger.info(f"Converted to snake case: {factory_graph_snake_case_dict}")
    factory_graph_snake_case = FactoryProductionGraph(**factory_graph_snake_case_dict)
    return create_factory_graph(factory_graph_snake_case)


@router.get("/api/v1/ml/factory-graphs/{id}")
async def read_factory_graph(id: str):
    return get_factory_graph_by_id(int(id))


# Testing endpoints
@router.get("/api/v1/ml/dependency-graphs/{id}/stage_output/{stage_id}")
async def read_dependency_graph_stage_output(id: int, stage_id: int):
    graph_data = get_factory_graph_by_id(int(id))
    return determine_dependency_subtree(graph_data.factory_graph, stage_id)

@router.get("/api/v1/ml/topological-order/{id}")
async def topological_order_graph_stage_output(id: int):
    graph_data = get_factory_graph_by_id(int(id))
    return topological_sort(graph_data.factory_graph)

@router.get("/api/v1/ml/classic-optimization-new/{id}", response_model=dict[int, float])
async def read_classic_optimization(id: str):
    numeric_id = int(id)
    graph_data = get_factory_graph_by_id(numeric_id)
    inventory, priorities = generate_data(graph_data.factory_graph)

    max_outputs = compute_max_outputs(graph_data.factory_graph, inventory, priorities)

    return max_outputs