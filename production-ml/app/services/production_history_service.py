

import logging

from fastapi import HTTPException

from app.config.database_connection import get_db
from app.types.production_history_model import FactoryProductionHistory
from app.utils.utils import convert_keys_to_strings, convert_string_keys_to_floats, deserialize_to_model


logger = logging.getLogger(__name__)

def create_production_history(history: FactoryProductionHistory):
    db = get_db()
    history_data = serialize_production_history(history)
    history_data['productionHistory']['dailyProductionRecords'] = convert_keys_to_strings(
        history_data['productionHistory']['dailyProductionRecords']
    )
    logger.info("Serialized data to insert: %s", history_data)

    history_data['_id'] = history_data['id']
    logger.info("Inserting with ID: %s", history_data['_id'])

    db.production_histories.insert_one(history_data)
    return {"message": "Production history added"}

def get_production_history(id: str):
    numeric_id = int(id)

    db = get_db()
    history_data = db.production_histories.find_one({"_id": numeric_id})
    if not history_data:
        logger.error(f"No history found for ID: {numeric_id}")
        raise HTTPException(status_code=404, detail="History not found")
    
    history_data['productionHistory']['dailyProductionRecords'] = convert_string_keys_to_floats(
        history_data['productionHistory']['dailyProductionRecords']
    )
    
    model_data = deserialize_to_model(history_data, FactoryProductionHistory)
    return model_data.model_dump()


def serialize_production_history(history: FactoryProductionHistory):
    history_data = history.model_dump(by_alias=True)

    daily_records = {}
    for key, record in history.production_history.daily_production_records.items():
        daily_records[str(key)] = record.model_dump()
    history_data['productionHistory']['dailyProductionRecords'] = daily_records
    return history_data