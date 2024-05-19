import logging

from fastapi import APIRouter

from app.ml.model.model import FactoryEnvironment, GNNModel, train
from app.services.factory_graph_service import get_factory_graph_by_id

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/train-model/{id}")
async def train_model(id: str):
    numeric_id = int(id)
    graph_data = get_factory_graph_by_id(numeric_id)

    in_feats = {
        'stage': 2,
        'input': 2,
        'output': 2,
    }
    hidden_size = {
        'stage': 50,
        'input': 50,
        'output': 50,
    }
    num_classes = {
        'stage': 6,
        'input': 8,
        'output': 7,
    }

    model = GNNModel(in_feats, hidden_size, num_classes)

    env = FactoryEnvironment(graph_data.factory_graph, model)

    train(model, env, 100, 1)
