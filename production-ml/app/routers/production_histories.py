from fastapi import APIRouter

from app.services.production_history_service import create_production_history, get_production_history
from app.types import FactoryProductionHistory


router = APIRouter()

@router.post("/production-histories/")
async def create_production_history_endpoint(history: FactoryProductionHistory):
    return create_production_history(history)

@router.get("/production-histories/{id}")
async def read_production_history(id: str):
    return get_production_history(id)