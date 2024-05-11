from pymongo import MongoClient
import logging

logging.basicConfig(level=logging.INFO)

class Database:
    client: MongoClient = None

def get_db():
    if not Database.client:
        Database.client = MongoClient("mongodb://mongodb:27017/")
        logging.info("Connecting to the database...")
    return Database.client["mydatabase"]