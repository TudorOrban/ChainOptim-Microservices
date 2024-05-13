from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Optional

from app.types.factory_inventory import FactoryInventoryItem
from app.utils.common import to_camel

class FactoryProductionHistory(BaseModel):
    id: int = Field(..., alias=to_camel('id'))
    factory_id: int = Field(..., alias=to_camel('factory_id'))
    created_at: datetime = Field(..., alias=to_camel('created_at'))
    updated_at: datetime = Field(..., alias=to_camel('updated_at'))
    production_history: 'ProductionHistory' = Field(..., alias=to_camel('production_history'))

    class Config:
        arbitrary_types_allowed = True
        populate_by_name = True
        from_attributes = True

class ProductionHistory(BaseModel):
    start_date: datetime = Field(..., alias=to_camel('start_date'))
    daily_production_records: Dict[str, 'DailyProductionRecord'] = Field(..., alias=to_camel('daily_production_records'))
    
class DailyProductionRecord(BaseModel):
    allocations: List['ResourceAllocation'] = Field(..., alias=to_camel('allocations'))
    results: List['AllocationResult'] = Field(..., alias=to_camel('results'))
    inventory: List['FactoryInventoryItem'] = Field(..., alias=to_camel('inventory'))
    duration_days: Optional[float] = Field(default=None, alias=to_camel('duration_days'))

class ResourceAllocation(BaseModel):
    stage_input_id: int = Field(..., alias=to_camel('stage_input_id'))
    factory_stage_id: int = Field(..., alias=to_camel('factory_stage_id'))
    stage_name: str = Field(..., alias=to_camel('stage_name'))
    component_id: int = Field(..., alias=to_camel('component_id'))
    component_name: str = Field(..., alias=to_camel('component_name'))
    allocator_inventory_item_id: int = Field(..., alias=to_camel('allocator_inventory_item_id'))
    allocated_amount: Optional[float] = Field(default=None, alias=to_camel('allocated_amount'))
    requested_amount: Optional[float] = Field(default=None, alias=to_camel('requested_amount'))
    actual_amount: Optional[float] = Field(default=None, alias=to_camel('actual_amount'))

class AllocationResult(BaseModel):
    stage_output_id: int = Field(..., alias=to_camel('stage_output_id'))
    factory_stage_id: int = Field(..., alias=to_camel('factory_stage_id'))
    stage_name: str = Field(..., alias=to_camel('stage_name'))
    component_id: int = Field(..., alias=to_camel('component_id'))
    component_name: str = Field(..., alias=to_camel('component_name'))
    resulted_amount: Optional[float] = Field(default=None, alias=to_camel('resulted_amount'))
    full_amount: Optional[float] = Field(default=None, alias=to_camel('full_amount'))
    actual_amount: Optional[float] = Field(default=None, alias=to_camel('actual_amount'))

