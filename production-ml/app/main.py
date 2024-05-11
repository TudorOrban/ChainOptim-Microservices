from dataclasses import asdict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

from app.database_connection import get_db
from app.types import FactoryProductionHistory
from app.utils import convert_float_keys_to_strings, convert_string_keys_to_floats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

db = get_db()



def serialize_production_history(history: FactoryProductionHistory):
    history_data = history.model_dump(by_alias=True)

    daily_records = {}
    for key, record in history.production_history.daily_production_records.items():
        daily_records[str(key)] = record.model_dump()  # Ensure each record is fully serialized
    history_data['productionHistory']['dailyProductionRecords'] = daily_records
    return history_data

@app.post("/production-histories/")
async def create_production_history(history: FactoryProductionHistory):
    history_data = serialize_production_history(history)
    history_data['productionHistory']['dailyProductionRecords'] = convert_float_keys_to_strings(
        history_data['productionHistory']['dailyProductionRecords']
    )
    logger.info("Serialized data to insert: %s", history_data)
    
    db.production_histories.insert_one(history_data)
    
    return {"message": "Production history added"}

@app.get("/production-histories/{id}")
async def read_production_history(id: str):
    history_data = db.production_histories.find_one({"_id": id})
    if not history_data:
        raise HTTPException(status_code=404, detail="History not found")
    history_data['productionHistory']['dailyProductionRecords'] = convert_string_keys_to_floats(
        history_data['productionHistory']['dailyProductionRecords']
    )

    history = FactoryProductionHistory(**history_data)
    
    return history.model_dump()
