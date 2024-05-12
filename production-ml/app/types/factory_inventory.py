
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from app.utils.common import to_camel


class FactoryInventoryItem(BaseModel):
    id: int = Field(..., alias=to_camel('id'))
    factory_id: int = Field(..., alias=to_camel('factory_id'))
    product: Optional['Product'] = Field(..., alias=to_camel('product'))
    component: Optional['Component'] = Field(..., alias=to_camel('component'))
    quantity: float = Field(..., alias=to_camel('quantity'))
    created_at: datetime = Field(..., alias=to_camel('created_at'))
    updated_at: datetime = Field(..., alias=to_camel('updated_at'))


class Product(BaseModel):
    id: int = Field(..., alias=to_camel('id'))
    name: str = Field(..., alias=to_camel('name'))
    unit: 'UnitOfMeasurement' = Field(..., alias=to_camel('unit'))

class Component(BaseModel):
    id: int = Field(..., alias=to_camel('id'))
    name: str = Field(..., alias=to_camel('name'))
    unit: 'UnitOfMeasurement' = Field(..., alias=to_camel('unit'))

class UnitOfMeasurement(BaseModel):
    id: int = Field(..., alias=to_camel('id'))
    name: str = Field(..., alias=to_camel('name'))