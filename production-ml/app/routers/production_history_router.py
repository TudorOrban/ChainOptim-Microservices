from fastapi import APIRouter

from app.services.production_history_service import create_production_history, get_production_history
from app.types.production_history_model import FactoryProductionHistory


router = APIRouter()

@router.post("/api/v1/ml/production-histories/")
async def create_production_history_endpoint(history: FactoryProductionHistory):
    return create_production_history(history)

@router.get("/api/v1/ml/production-histories/{id}")
async def read_production_history(id: str):
    return get_production_history(id)