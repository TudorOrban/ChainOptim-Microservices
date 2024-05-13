import logging

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from app.config.database_connection import get_db
from app.ml.data.data_generator import generate_data
from app.ml.model.classic_optimizer import determine_optimal_distribution
from app.types.factory_graph import FactoryProductionGraph
from app.utils.utils import convert_string_keys_to_floats

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/classic-optimization/{id}", response_model=dict[int, float])
async def read_classic_optimization(id: str):
    numeric_id = int(id)
    db = get_db()
    document = db.factory_production_graphs.find_one({"_id": numeric_id})
    if document is None:
        logger.error(f"No graph found for ID: {numeric_id}")
        raise HTTPException(status_code=404, detail="Graph not found")
    
    try:
        document['factoryGraph']['nodes'] = convert_string_keys_to_floats(
            document['factoryGraph']['nodes']
        )
        document['factoryGraph']['adjList'] = convert_string_keys_to_floats(
            document['factoryGraph']['adjList']
        )
        graph_data = FactoryProductionGraph(**document)
        inventory, priorities = generate_data(graph_data.factory_graph)
        optimal_distribution = determine_optimal_distribution(graph_data.factory_graph, inventory, priorities)

        return optimal_distribution

    except ValidationError as e:
        logger.error(f"Validation error for the document with ID {numeric_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Data validation error")
    