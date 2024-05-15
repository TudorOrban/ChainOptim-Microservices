import logging
from typing import List
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn
import torch
from torch import Tensor

from app.ml.data.data_generator import generate_data
from app.ml.model.graph import build_heterogeneous_graph
from app.ml.model.new_resource_distributor import compute_max_outputs_new
from app.types.factory_graph import FactoryGraph
from app.types.factory_inventory import FactoryInventoryItem

logger = logging.getLogger(__name__)

class GNNModel(nn.Module):
    def __init__(self, in_feats: int, hidden_size: int, num_classes: int):
        super(GNNModel, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_size)
        self.conv2 = dglnn.GraphConv(hidden_size, num_classes)

    def forward(self, g: dgl.DGLGraph) -> Tensor:
        x = g.ndata['features']
        
        if g.canonical_etypes is None:
            raise ValueError("Graph must have canonical edge types")
        
        for etype in g.canonical_etypes:
            if 'conv1' in self.__dict__ and isinstance(self.conv1, dglnn.GraphConv):
                x = torch.relu(self.conv1(g[etype], x))
            if 'conv2' in self.__dict__ and isinstance(self.conv2, dglnn.GraphConv):
                x = self.conv2(g[etype], x)

        print("Output from model's forward method:", type(x), x.shape if isinstance(x, Tensor) else "Not a tensor")
        return x
    
def inventory_to_tensor(inventory: List[FactoryInventoryItem]) -> Tensor:
    quantities = [item.quantity for item in inventory]
    return torch.tensor(quantities, dtype=torch.float32).unsqueeze(1)

def priorities_to_tensor(priorities: dict[int, float]) -> Tensor:
    priority_values = list(priorities.values())
    return torch.tensor(priority_values, dtype=torch.float32).unsqueeze(1)


class FactoryEnvironment:
    def __init__(self, factory_graph: FactoryGraph, model: GNNModel):
        self.factory_graph = factory_graph
        self.model = model
        self.inventory, self.priorities = generate_data(self.factory_graph)
        self.inventory_tensor = inventory_to_tensor(self.inventory)
        self.priorities_tensor = priorities_to_tensor(self.priorities)
        logger.info(f"Inventory tensor: {self.inventory_tensor}")
        self.graph = build_heterogeneous_graph(factory_graph)
        logger.info(f"Graph: {self.graph}")
    
    def compute_reward(self, action: Tensor) -> float:
        optimal_distribution = compute_max_outputs_new(self.factory_graph, self.inventory, self.priorities)
        reward = -torch.mean((action - torch.tensor(list(optimal_distribution.values()))).pow(2)).item()
        return reward

    def reset(self):
        self.inventory, self.priorities = generate_data(self.factory_graph)
        self.inventory_tensor = inventory_to_tensor(self.inventory)
        self.priorities_tensor = priorities_to_tensor(self.priorities)
        return {'graph': self.graph, 'inventory': self.inventory_tensor, 'priorities': self.priorities_tensor}

    def step(self, action: Tensor):
        reward = self.compute_reward(action)
        done = True  
        return {'graph': self.graph, 'inventory': self.inventory_tensor, 'priorities': self.priorities_tensor}, reward, done
    

def train(model: GNNModel, env: FactoryEnvironment, episodes: int):
    for episode in range(episodes):
        state = env.reset()
        total_reward: float = 0
        done: bool = False
        while not done:
            graph = state['graph']
            action = model(graph)
            print("Action shape:", action.shape)
            print("Action sample:", action[:5])
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
        print(f"Episode: {episode}, Total reward: {total_reward}")