import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn
import torch



class GNNModel(nn.Module):
    def __init__(self, in_feats: int, hidden_size: int, num_classes: int):
        super(GNNModel, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_size)
        self.conv2 = dglnn.GraphConv(hidden_size, num_classes)

    def forward(self, g: dgl.DGLGraph, inventory_levels: torch.Tensor, priorities: torch.Tensor) -> torch.Tensor:
        x = torch.cat([inventory_levels, priorities], dim=1)
        x = torch.relu(self.conv1(g, x))
        x = self.conv2(g, x)
        return x