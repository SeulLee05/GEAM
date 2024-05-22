import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import NNConv, global_mean_pool
from torch_scatter import scatter_mean, scatter_add, scatter_std


class GatherModel(nn.Module):
    """
    MPNN from `Neural Message Passing for Quantum Chemistry <https://arxiv.org/abs/1704.01212>`
    """
    def __init__(self,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=42,
                 num_step_message_passing=3,
                 dropout=0.0):
        super().__init__()

        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.message_layer = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim), nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))
        self.conv = NNConv(in_channels=node_hidden_dim,
                           out_channels=node_hidden_dim,
                           nn=edge_network,
                           aggr='add',
                           root_weight=True
                           )
        self.dropout = dropout

    def forward(self, g):
        init = g.x.clone()
        out = F.relu(self.lin0(g.x))
        for i in range(self.num_step_message_passing):
            if len(g.edge_attr) != 0:
                m = torch.relu(self.conv(out, g.edge_index, g.edge_attr))
            else:
                m = torch.relu(self.conv.bias + out)
            out = self.message_layer(torch.cat([m, out], dim=1))
        return out + init


class FGIB(nn.Module):
    def __init__(self,
                device,
                node_input_dim=44,
                edge_input_dim=10,
                node_hidden_dim=44):
        super().__init__()

        self.device = device

        self.gather = GatherModel(node_input_dim, edge_input_dim,
                                  node_hidden_dim, edge_input_dim)

        self.compressor = nn.Sequential(
            nn.Linear(node_hidden_dim, node_hidden_dim),
            nn.BatchNorm1d(node_hidden_dim),
            nn.ReLU(),
            nn.Linear(node_hidden_dim, 1),
            nn.Sigmoid())
        
        self.predictor = nn.Sequential(
            nn.Linear(node_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid())
        
        self.mse_loss = torch.nn.MSELoss()

        self.init_model()
    
    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, graph, get_w=False):
        node_features = F.normalize(self.gather(graph), dim=1)
        frag_features = global_mean_pool(node_features, graph.node2frag_batch)
        lambda_pos = w = self.compressor(frag_features).squeeze()
        
        if get_w:
            return w.cpu()
        
        lambda_pos = lambda_pos.reshape(-1, 1)
        lambda_neg = 1 - lambda_pos
        preserve_rate = (w > 0.5).float().mean()
        
        static_feature = frag_features.clone().detach()
        frag_feature_mean = scatter_mean(static_feature, graph.frag2graph_batch, dim=0)[graph.frag2graph_batch]
        frag_feature_std = scatter_std(static_feature, graph.frag2graph_batch, dim=0)[graph.frag2graph_batch]
        
        noisy_frag_feature_mean = lambda_pos * frag_features + lambda_neg * frag_feature_mean
        noisy_frag_feature_std = lambda_neg * frag_feature_std

        noisy_frag_feature = noisy_frag_feature_mean + torch.randn_like(noisy_frag_feature_mean) * noisy_frag_feature_std
        noisy_subgraph = global_mean_pool(noisy_frag_feature, graph.frag2graph_batch)
        pred = self.predictor(noisy_subgraph)

        epsilon = 1e-7
        KL_tensor = 0.5 * scatter_add(((noisy_frag_feature_std ** 2) / (frag_feature_std + epsilon) ** 2).mean(dim=1), graph.frag2graph_batch).reshape(-1, 1) + \
                    scatter_add((((noisy_frag_feature_mean - frag_feature_mean) / (frag_feature_std + epsilon)) ** 2), graph.frag2graph_batch, dim=0)
        KL_loss = KL_tensor.mean()
        
        return pred, KL_loss, preserve_rate
