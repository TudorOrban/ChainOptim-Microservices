import logging
from typing import Dict
import dgl
import dgl.nn as dglnn
import torch
from torch import Tensor

from app.ml.data.data_generator import generate_data
from app.ml.model.graph_builder import build_heterogeneous_graph
from app.ml.model.resource_distributor import compute_max_outputs
from app.ml.pipeline.input_pipeline import inventory_to_tensor, priorities_to_tensor
from app.types.factory_graph import FactoryGraph

logger = logging.getLogger(__name__)


class GNNModel(torch.nn.Module):
    def __init__(self, in_feats: Dict[str, int], hidden_size: Dict[str, int], num_classes: Dict[str, int]):
        super(GNNModel, self).__init__()
        self.convs = torch.nn.ModuleDict({
            key: dglnn.GraphConv(in_feats[key], hidden_size[key], allow_zero_in_degree=True) # type: ignore
            for key in in_feats.keys()
        })
        self.convs2 = torch.nn.ModuleDict({
            key: dglnn.GraphConv(hidden_size[key], num_classes[key], allow_zero_in_degree=True) # type: ignore
            for key in hidden_size.keys()
        })
        self.num_classes = num_classes

    def forward(self, g: dgl.DGLGraph) -> Dict[str, torch.Tensor]:
        if 'features' not in g.ndata:
            raise ValueError("Graph must have node features under 'features'")
        if not g.canonical_etypes:
            raise ValueError("Graph must have canonical edge types")

        results = {}
        for etype in g.canonical_etypes:
            src_type, _, _ = etype

            # Create subgraph for the current edge type
            edge_subgraph = dgl.edge_type_subgraph(g, [etype])

            if src_type not in edge_subgraph.ndata['features']:
                continue

            src_features: torch.Tensor = edge_subgraph.nodes[src_type].data['features']
            if src_features.nelement() == 0:
                continue

            print("Src features before conv1:", src_features.shape)
            src_features = torch.relu(self.convs[src_type](edge_subgraph, src_features))
            print("Src features after conv1:", src_features.shape)

            # Pad to ensure feature tensor shape matches the number of nodes
            num_nodes = edge_subgraph.num_nodes(src_type)
            if src_features.shape[0] < num_nodes:
                padded_features = torch.zeros((num_nodes, src_features.shape[1]),
                                              device=src_features.device, dtype=src_features.dtype)
                in_degrees = edge_subgraph.in_degrees(etype=etype)
                nodes_with_edges = (in_degrees > 0).nonzero(as_tuple=True)[0]
                padded_features[nodes_with_edges] = src_features
                src_features = padded_features

            print("Src features after padding:", src_features.shape)
            
            try:
                src_features = self.convs2[src_type](edge_subgraph, src_features)
            except RuntimeError as e:
                print("Error during graph convolution:")
                print(f"Local graph structure: Nodes={edge_subgraph.number_of_nodes()}, Edges={edge_subgraph.number_of_edges()}")
                print(f"src_features size: {src_features.size()}")
                raise RuntimeError("Failed during convolution operation") from e

            results[src_type] = src_features

        return results if results else {ntype: torch.zeros((self.num_classes[ntype],), dtype=torch.float32) for ntype in self.num_classes}


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

        # Align the sizes by trimming or padding the optimal values tensor
        if action.shape[0] != optimal_values_tensor.shape[0]:
            logger.info("Mismatch in action and optimal values tensor sizes.")
            
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
