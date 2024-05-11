from fastapi import FastAPI
from pymongo import MongoClient
import logging

from app.database_connection import get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
def read_root():
    db = get_db()  # This will log connection info only on the first call due to the logic in get_db
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int):
    db = get_db()
    collection = db["production"]
    item = collection.find_one({"item_id": item_id})
    return {"item_name": item['name']} if item else {"error": "Item not found"}