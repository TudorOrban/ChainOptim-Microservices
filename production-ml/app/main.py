from fastapi import FastAPI, HTTPException
import logging

from app.database_connection import get_db
from app.types import FactoryProductionHistory
from app.utils import convert_float_keys_to_strings, convert_string_keys_to_floats
from app.routers import production_histories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

db = get_db()

app.include_router(production_histories.router)