from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from app.utils.common import to_camel

class FactoryProductionGraph(BaseModel):
    id: int
    factory_id: int
    created_at: datetime
    updated_at: datetime
    factory_graph: 'FactoryGraph'

    class Config:
        from_attributes = True

class FactoryGraph(BaseModel):
    nodes: Dict[int, 'StageNode'] # stage_id -> StageNode
    adj_list: Dict[int, List['Edge']] # stage_id -> List[Edge]

class StageNode(BaseModel):
    small_stage: 'SmallStage'
    number_of_steps_capacity: Optional[float] = None
    per_duration: Optional[float] = None
    minimum_required_capacity: Optional[float] = None
    priority: Optional[float] = None
    allocation_capacity_ratio: Optional[float] = None

class Edge(BaseModel):
    incoming_factory_stage_id: int
    incoming_stage_output_id: int
    outgoing_factory_stage_id: int
    outgoing_stage_input_id: int
    
    def __hash__(self):
        return hash((self.incoming_factory_stage_id, self.incoming_stage_output_id,
                     self.outgoing_factory_stage_id, self.outgoing_stage_input_id))

    def __eq__(self, other):
        if isinstance(other, Edge):
            return (self.incoming_factory_stage_id == other.incoming_factory_stage_id and
                    self.incoming_stage_output_id == other.incoming_stage_output_id and
                    self.outgoing_factory_stage_id == other.outgoing_factory_stage_id and
                    self.outgoing_stage_input_id == other.outgoing_stage_input_id)
        return False

class SmallStage(BaseModel):
    id: int
    stage_name: str
    stage_inputs: List['SmallStageInput']
    stage_outputs: List['SmallStageOutput']

class SmallStageInput(BaseModel):
    id: int
    component_id: int
    component_name: str 
    quantity_per_stage: Optional[float] = None
    allocated_quantity: Optional[float] = None
    requested_quantity: Optional[float] = None

class SmallStageOutput(BaseModel):
    id: int
    component_id: Optional[int] = None
    component_name: Optional[str] = None
    product_id: Optional[int] = None
    product_name: Optional[str] = None
    quantity_per_stage: Optional[float] = None
    expected_output_per_allocation: Optional[float] = None
    output_per_request: Optional[float] = None


## Camel case types
class CamelCaseFactoryProductionGraph(BaseModel):
    id: int
    factoryId: int
    createdAt: datetime
    updatedAt: datetime
    factoryGraph: 'CamelCaseFactoryGraph'

class CamelCaseFactoryGraph(BaseModel):
    nodes: Dict[int, 'CamelCaseStageNode']
    adjList: Dict[int, List['CamelCaseEdge']]

class CamelCaseStageNode(BaseModel):
    smallStage: 'CamelCaseSmallStage'
    numberOfStepsCapacity: Optional[float] = None
    perDuration: Optional[float] = None
    minimumRequiredCapacity: Optional[float] = None
    priority: Optional[float] = None
    allocationCapacityRatio: Optional[float] = None

class CamelCaseEdge(BaseModel):
    incomingFactoryStageId: int
    incomingStageOutputId: int
    outgoingFactoryStageId: int
    outgoingStageInputId: int

class CamelCaseSmallStage(BaseModel):
    id: int
    stageName: str
    stageInputs: List['CamelCaseSmallStageInput']
    stageOutputs: List['CamelCaseSmallStageOutput']

class CamelCaseSmallStageInput(BaseModel):
    id: int
    componentId: int
    componentName: str 
    quantityPerStage: Optional[float] = None
    allocatedQuantity: Optional[float] = None
    requestedQuantity: Optional[float] = None

class CamelCaseSmallStageOutput(BaseModel):
    id: int
    componentId: Optional[int] = None
    componentName: Optional[str] = None
    productId: Optional[int] = None
    productName: Optional[str] = None
    quantityPerStage: Optional[float] = None
    expectedOutputPerAllocation: Optional[float] = None
    outputPerRequest: Optional[float] = None