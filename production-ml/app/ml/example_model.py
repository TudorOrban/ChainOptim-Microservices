import torch.nn as nn


class BasicGNN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(BasicGNN, self).__init__()
        self.conv1 = nn.GraphConv(in_feats, hidden_size)
        self.conv2 = nn.GraphConv(hidden_size, num_classes)

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = self.conv2(g, x)
        return x
    
model = BasicGNN(5, 10, 2)