
import logging

from fastapi import HTTPException
from pydantic import ValidationError
from app.config.database_connection import get_db

from app.ml.data.data_generator import generate_data
from app.ml.model.classic_optimizer import determine_optimal_distribution
from app.ml.pipeline.input_pipeline import apply_unified_id_mapping, unify_ids
from app.types.factory_graph import FactoryProductionGraph
from app.utils.utils import convert_keys_to_strings, convert_string_keys_to_floats, deserialize_to_model

logger = logging.getLogger(__name__)

def create_factory_graph(production_graph: FactoryProductionGraph):
    db = get_db() # type: ignore
    id_map = unify_ids(production_graph.factory_graph)
    production_graph.factory_graph = apply_unified_id_mapping(production_graph.factory_graph, id_map)
    graph_data = production_graph.model_dump(by_alias=True)
    graph_data['factory_graph']['nodes'] = convert_keys_to_strings(
        graph_data['factory_graph']['nodes']
    )
    graph_data['factory_graph']['adj_list'] = convert_keys_to_strings(
        graph_data['factory_graph']['adj_list']
    )
    logger.info("Serialized data to insert: %s", graph_data)


    graph_data['_id'] = graph_data['id']
    logger.info("Inserting with ID: %s", graph_data['_id'])

    db.factory_production_graphs.insert_one(graph_data)
    return {"message": "Factory graph added"}


def get_factory_graph_by_id(id: int) -> FactoryProductionGraph:
    db = get_db()
    document = db.factory_production_graphs.find_one({"_id": id})
    if document is None:
        logger.error(f"No graph found for ID: {id}")
        raise HTTPException(status_code=404, detail="Graph not found")
    
    try:
        graph_data = FactoryProductionGraph(**document)
        return graph_data
    except ValidationError as e:
        logger.error(f"Validation error for the document with ID {id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Data validation error")


def serialize_factory_graph(production_graph: FactoryProductionGraph):
    production_graph_data = production_graph.model_dump(by_alias=True)

    nodes = {}
    for key, node in production_graph.factory_graph.nodes.items():
        nodes[str(key)] = node.model_dump()

    production_graph_data['factoryGraph']['nodes'] = nodes
    return production_graph_data