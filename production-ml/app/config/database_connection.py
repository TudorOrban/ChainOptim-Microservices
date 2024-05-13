from typing import Optional
from pymongo import MongoClient
import logging

logging.basicConfig(level=logging.INFO)

class Database:
    client: Optional[MongoClient] = None

def get_db():
    if Database.client is None:
        Database.client = MongoClient("mongodb://mongodb:27017/")
        logging.info("Connecting to the database...")
    return Database.client["mydatabase"]
