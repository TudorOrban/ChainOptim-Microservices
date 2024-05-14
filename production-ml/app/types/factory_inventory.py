
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from app.utils.common import to_camel


class FactoryInventoryItem(BaseModel):
    id: int
    factory_id: int
    product: Optional['Product'] = None
    component: Optional['Component'] = None
    quantity: float
    created_at: datetime
    updated_at: datetime


class Product(BaseModel):
    id: int
    name: str
    unit: 'UnitOfMeasurement'

class Component(BaseModel):
    id: int
    name: str
    unit: 'UnitOfMeasurement'

class UnitOfMeasurement(BaseModel):
    id: int
    name: str