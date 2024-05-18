import logging
from typing import Dict
import torch.nn as nn
import dgl
import dgl.nn as dglnn
import torch
from torch import Tensor
import dgl.function as fn

from app.ml.data.data_generator import generate_data
from app.ml.model.graph_builder import build_heterogeneous_graph
from app.ml.model.resource_distributor import compute_max_outputs
from app.ml.pipeline.input_pipeline import inventory_to_tensor, priorities_to_tensor
from app.types.factory_graph import FactoryGraph

logger = logging.getLogger(__name__)

# class CustomGraphConv(nn.Module):
#     def __init__(self, in_feats, out_feats):
#         super(CustomGraphConv, self).__init__()
#         self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
#         nn.init.xavier_uniform_(self.weight)

#     def forward(self, graph, feature):
#         with graph.local_scope():
#             graph.ndata['h'] = torch.matmul(feature, self.weight)
#             graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
#             return graph.ndata['h']




class GNNModel(torch.nn.Module):
    def __init__(self, in_feats: Dict[str, int], hidden_size: Dict[str, int], num_classes: Dict[str, int]):
        super(GNNModel, self).__init__()
        # Initialize GraphConv layers for each node type separately
        self.convs = {ntype: dglnn.GraphConv(in_feats[ntype], hidden_size[ntype], allow_zero_in_degree=True) for ntype in in_feats}
        self.convs2 = {ntype: dglnn.GraphConv(hidden_size[ntype], num_classes[ntype], allow_zero_in_degree=True) for ntype in hidden_size}

    def forward(self, g: dgl.DGLGraph) -> Dict[str, torch.Tensor]:
        if not g.ntypes:
            raise ValueError("Graph must have node types")
        
        if not g.canonical_etypes:
            raise ValueError("Graph does not contain any canonical edge types")
        
        h_dict = {}
        for ntype in g.ntypes:
            if ntype in g.ndata['features']:
                h = torch.relu(self.convs[ntype](g, g.ndata['features'][ntype]))
                h = self.convs2[ntype](g, h)
                h_dict[ntype] = h

        if not h_dict:
            raise RuntimeError("No features processed, check node types and features data")

        return h_dict


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
        optimal_values_tensor = torch.tensor(list(optimal_distribution.values()), dtype=torch.float32).to(action.device)

        logger.info(f"Action tensor shape: {action.shape}")
        logger.info(f"Optimal values tensor shape: {optimal_values_tensor.shape}")

        if action.shape[0] != optimal_values_tensor.shape[0]:
            logger.info("Mismatch in action and optimal values tensor sizes.")
            
            # Align the sizes by trimming or padding the optimal values tensor
            if action.shape[0] < optimal_values_tensor.shape[0]:
                optimal_values_tensor = optimal_values_tensor[:action.shape[0]]
            else:
                padding = torch.zeros(action.shape[0] - optimal_values_tensor.shape[0], device=optimal_values_tensor.device)
                optimal_values_tensor = torch.cat((optimal_values_tensor, padding), dim=0)

            logger.info(f"Aligned optimal values tensor shape: {optimal_values_tensor.shape}")

        # Reduce action tensor dimensions if necessary
        if action.dim() > 1:
            action = action.mean(dim=1) 
            logger.info(f"Reduced action tensor shape: {action.shape}")

        reward = -torch.mean((action - optimal_values_tensor).pow(2)).item()
        return reward


    def reset_environment(self, new_data=True):
        if new_data:
            self.inventory, self.priorities = generate_data(self.factory_graph)
        self.inventory_tensor = inventory_to_tensor(self.inventory)
        self.priorities_tensor = priorities_to_tensor(self.priorities)
        return {'graph': self.graph, 'inventory': self.inventory_tensor, 'priorities': self.priorities_tensor}

    
    def step(self, action: Tensor, update_data=False):
        reward = self.compute_reward(action)
        if update_data:
            self.reset_environment(new_data=False)
        done = False
        return {'graph': self.graph, 'inventory': self.inventory_tensor, 'priorities': self.priorities_tensor}, reward, done

def train(model: GNNModel, env: FactoryEnvironment, episodes: int, steps_per_episode: int):
    for episode in range(episodes):
        env.reset_environment(new_data=True) # Start with new data each episode
        total_reward = 0
        for _ in range(steps_per_episode):
            state = {'graph': env.graph, 'inventory': env.inventory_tensor, 'priorities': env.priorities_tensor}
            action = model(state['graph'])
            _, reward, done = env.step(action, update_data=True)
            total_reward += reward
            if done:
                break
        print(f"Episode: {episode}, Total reward: {total_reward}")
