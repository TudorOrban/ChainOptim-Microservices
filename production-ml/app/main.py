from dataclasses import asdict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

from app.database_connection import get_db
from app.types import FactoryProductionHistory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

db = get_db()


@app.post("/production-histories/")
def create_production_history(history: FactoryProductionHistory):
    # Assuming MongoDB insertion is setup
    db.production_histories.insert_one(history.dict(by_alias=True))
    return {"message": "Production history added", "data": history}

@app.get("/production-histories/{id}")
def read_production_history(id: str):
    history_data = db.production_histories.find_one({"_id": id})
    if not history_data:
        raise HTTPException(status_code=404, detail="History not found")
    history = FactoryProductionHistory(**history_data)
    return history