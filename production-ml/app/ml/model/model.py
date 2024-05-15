import logging
from typing import List
import torch.nn as nn
import dgl
import dgl.nn as dglnn
import torch
from torch import Tensor

from app.ml.data.data_generator import generate_data
from app.ml.model.graph import build_heterogeneous_graph
from app.ml.model.resource_distributor import compute_max_outputs
from app.types.factory_graph import FactoryGraph
from app.types.factory_inventory import FactoryInventoryItem

logger = logging.getLogger(__name__)

class GNNModel(nn.Module):
    def __init__(self, in_feats: int, hidden_size: int, num_classes: int):
        super(GNNModel, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_size, allow_zero_in_degree=True) # type: ignore
        self.conv2 = dglnn.GraphConv(hidden_size, num_classes, allow_zero_in_degree=True) # type: ignore
        self.num_classes = num_classes

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        if 'features' not in g.ndata:
            raise ValueError("Graph must have node features under 'features'")
        if not g.canonical_etypes:
            raise ValueError("Graph must have canonical edge types")

        results = []
        for etype in g.canonical_etypes:
            local_g = g[etype]
            srctype, _, _ = etype

            if 'features' in local_g.ndata[srctype]:
                src_features = local_g.ndata['features'][srctype]
                in_degrees = local_g.in_degrees().numpy()
                mask = in_degrees > 0

                if src_features.shape[0] == mask.size:
                    filtered_features = src_features[mask]
                    if filtered_features.shape[0] > 0:
                        filtered_features = torch.relu(self.conv1(local_g, filtered_features))
                        results.append(self.conv2(local_g, filtered_features))

        if results:
            x = torch.mean(torch.stack(results), dim=0)
        else:
            x = torch.zeros((self.num_classes,), dtype=torch.float32)

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
        self.graph = build_heterogeneous_graph(factory_graph)
    
    def compute_reward(self, action: Tensor) -> float:
        optimal_distribution = compute_max_outputs(self.factory_graph, self.inventory, self.priorities)
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