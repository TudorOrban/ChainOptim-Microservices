from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from app.utils.common import to_camel

class FactoryProductionGraph(BaseModel):
    id: int = Field(..., alias=to_camel('id'))
    factory_id: int = Field(..., alias=to_camel('factory_id'))
    created_at: datetime = Field(..., alias=to_camel('created_at'))
    updated_at: datetime = Field(..., alias=to_camel('updated_at'))
    factory_graph: 'FactoryGraph' = Field(..., alias=to_camel('factory_graph'))

class FactoryGraph(BaseModel):
    nodes: Dict[int, 'StageNode'] = Field(..., alias=to_camel('nodes'))
    adj_list: Dict[int, List['Edge']] = Field(..., alias=to_camel('adj_list'))
    pipeline_priority: Optional[float] = Field(default=None, alias=to_camel('pipeline_priority'))

class StageNode(BaseModel):
    small_stage: 'SmallStage' = Field(..., alias=to_camel('small_stage'))
    number_of_steps_capacity: Optional[float] = Field(default=None, alias=to_camel('number_of_steps_capacity'))
    per_duration: Optional[float] = Field(default=None, alias=to_camel('per_duration'))
    minimum_required_capacity: Optional[float] = Field(default=None, alias=to_camel('minimum_required_capacity'))
    priority: Optional[float] = Field(default=None, alias=to_camel('priority'))
    allocation_capacity_ratio: Optional[float] = Field(default=None, alias=to_camel('allocation_capacity_ratio'))

class Edge(BaseModel):
    incoming_factory_stage_id: int = Field(..., alias=to_camel('incoming_factory_stage_id'))
    incoming_stage_output_id: int = Field(..., alias=to_camel('incoming_stage_output_id'))
    outgoing_factory_stage_id: int = Field(..., alias=to_camel('outgoing_factory_stage_id'))
    outgoing_stage_input_id: int = Field(..., alias=to_camel('outgoing_stage_input_id'))

class SmallStage(BaseModel):
    id: int = Field(..., alias=to_camel('id'))
    stage_name: str = Field(..., alias=to_camel('stage_name'))
    stage_inputs: List['SmallStageInput'] = Field(..., alias=to_camel('stage_inputs'))
    stage_outputs: List['SmallStageOutput'] = Field(..., alias=to_camel('stage_outputs'))

class SmallStageInput(BaseModel):
    id: int = Field(..., alias=to_camel('id'))
    component_id: int = Field(..., alias=to_camel('component_id'))
    component_name: str = Field(..., alias=to_camel('component_name'))
    quantity_per_stage: Optional[float] = Field(default=None, alias=to_camel('quantity_per_stage'))
    allocated_quantity: Optional[float] = Field(default=None, alias=to_camel('allocated_quantity'))
    requested_quantity: Optional[float] = Field(default=None, alias=to_camel('requested_quantity'))

class SmallStageOutput(BaseModel):
    id: int = Field(..., alias=to_camel('id'))
    component_id: Optional[int] = Field(..., alias=to_camel('component_id'))
    component_name: Optional[str] = Field(..., alias=to_camel('component_name'))
    product_id: Optional[int] = Field(..., alias=to_camel('product_id'))
    product_name: Optional[str] = Field(..., alias=to_camel('product_name'))
    quantity_per_stage: Optional[float] = Field(default=None, alias=to_camel('quantity_per_stage'))
    expected_output_per_allocation: Optional[float] = Field(default=None, alias=to_camel('expected_output_per_allocation'))
    output_per_request: Optional[float] = Field(default=None, alias=to_camel('output_per_request'))
