

from fastapi import APIRouter

from app.services.factory_graph_service import create_factory_graph, get_factory_graph
from app.types.factory_graph import FactoryProductionGraph


router = APIRouter()

@router.post("/factory-graphs/")
async def create_factory_graph_endpoint(factory_graph: FactoryProductionGraph):
    return create_factory_graph(factory_graph)

@router.get("/factory-graphs/{id}")
async def read_factory_graph(id: str):
    return get_factory_graph(id)