import logging

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from app.config.database_connection import get_db
from app.ml.data.data_generator import generate_data
# from app.ml.model.classic_optimizer import determine_optimal_distribution
from app.ml.model.model import FactoryEnvironment, GNNModel, train
from app.services.factory_graph_service import get_factory_graph_by_id
from app.types.factory_graph import FactoryProductionGraph
from app.utils.utils import convert_string_keys_to_floats

logger = logging.getLogger(__name__)

router = APIRouter()


# @router.get("/classic-optimization/{id}", response_model=dict[int, float])
# async def read_classic_optimization(id: str):
#     numeric_id = int(id)
#     graph_data = get_factory_graph_by_id(numeric_id)
#     inventory, priorities = generate_data(graph_data.factory_graph)
#     optimal_distribution = determine_optimal_distribution(graph_data.factory_graph, inventory, priorities)

#     return optimal_distribution

@router.get("/train-model/{id}")
async def train_model(id: str):
    numeric_id = int(id)
    graph_data = get_factory_graph_by_id(numeric_id)

    in_feats = 2
    hidden_size = 50
    num_classes = 5

    model = GNNModel(in_feats, hidden_size, num_classes)

    env = FactoryEnvironment(graph_data.factory_graph, model)

    train(model, env, 50)
