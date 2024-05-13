
import logging

from fastapi import HTTPException
from app.config.database_connection import get_db

from app.types.factory_graph import FactoryProductionGraph
from app.utils.utils import convert_keys_to_strings, convert_string_keys_to_floats, deserialize_to_model

logger = logging.getLogger(__name__)

def create_factory_graph(production_graph: FactoryProductionGraph):
    db = get_db() # type: ignore
    graph_data = serialize_factory_graph(production_graph)
    graph_data['factoryGraph']['nodes'] = convert_keys_to_strings(
        graph_data['factoryGraph']['nodes']
    )
    graph_data['factoryGraph']['adjList'] = convert_keys_to_strings(
        graph_data['factoryGraph']['adjList']
    )
    logger.info("Serialized data to insert: %s", graph_data)

    graph_data['_id'] = graph_data['id']
    logger.info("Inserting with ID: %s", graph_data['_id'])

    db.factory_production_graphs.insert_one(graph_data)
    return {"message": "Factory graph added"}

def get_factory_graph(id: str) -> dict:
    numeric_id = int(id)

    db = get_db() # type: ignore
    graph_data = db.factory_production_graphs.find_one({"_id": numeric_id})
    if not graph_data:
        logger.error(f"No graph found for ID: {numeric_id}")
        raise HTTPException(status_code=404, detail="Graph not found")

    graph_data['factoryGraph']['nodes'] = convert_string_keys_to_floats(
        graph_data['factoryGraph']['nodes']
    )
    graph_data['factoryGraph']['adjList'] = convert_string_keys_to_floats(
        graph_data['factoryGraph']['adjList']
    )

    model_data = deserialize_to_model(graph_data, FactoryProductionGraph)
    return model_data.model_dump()



def serialize_factory_graph(production_graph: FactoryProductionGraph):
    production_graph_data = production_graph.model_dump(by_alias=True)

    nodes = {}
    for key, node in production_graph.factory_graph.nodes.items():
        nodes[str(key)] = node.model_dump()

    production_graph_data['factoryGraph']['nodes'] = nodes
    return production_graph_data