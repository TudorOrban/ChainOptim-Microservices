from typing import Tuple
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn
import torch
from torch import Tensor

from app.types.factory_graph import FactoryGraph



class GNNModel(nn.Module):
    def __init__(self, in_feats: int, hidden_size: int, num_classes: int):
        super(GNNModel, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_size)
        self.conv2 = dglnn.GraphConv(hidden_size, num_classes)

    def forward(self, g: dgl.DGLGraph, inventory_levels: Tensor, priorities: Tensor) -> Tensor:
        x = torch.cat([inventory_levels, priorities], dim=1)
        x = torch.relu(self.conv1(g, x))
        x = self.conv2(g, x)
        return x
    


class FactoryEnvironment:
    def __init__(self, factory_graph: FactoryGraph, model: GNNModel):
        self.factory_graph = factory_graph
        self.model = model
        self.state = torch.zeros((self.factory_graph.nodes.__len__(), 5))

    def reset(self) -> Tensor:
        self.state = self.initialize_state()
        return self.state
    
    def step(self, action: Tensor) -> Tuple[Tensor, float, bool]:
        next_state, reward, done = self.apply_action(action)
        return next_state, reward, done
    
    def initialize_state(self) -> Tensor:
        return torch.zeros((self.factory_graph.nodes.__len__(), 5))

    def apply_action(self, action: Tensor) -> Tuple[Tensor, float, bool]:
        reward = -torch.sum(action).item()
        return self.state, reward, False
    

def train(model: GNNModel, env: FactoryEnvironment, episodes: int):
    for episode in range(episodes):
        state: Tensor = env.reset()
        total_reward: float = 0
        done: bool = False
        while not done:
            state_tensor = torch.FloatTensor(state)
            action = model(state_tensor)
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
        print(f"Episode: {episode}, Total reward: {total_reward}")