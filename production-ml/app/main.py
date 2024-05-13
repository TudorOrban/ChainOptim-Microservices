from fastapi import FastAPI
import logging

from app.config.database_connection import get_db
from app.routers import production_history_router, factory_graph_router, ml_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

db = get_db() # type: ignore

app.include_router(production_history_router.router)
app.include_router(factory_graph_router.router)
app.include_router(ml_router.router)