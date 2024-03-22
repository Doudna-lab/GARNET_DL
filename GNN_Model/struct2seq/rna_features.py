from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy

from matplotlib import pyplot as plt

from .self_attention import gather_edges, gather_nodes, Normalize

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, period_range=[2,1000]):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.period_range = period_range 

    def forward(self, E_idx):
        # i-j
        device = E_idx.device
        N_batch = E_idx.size(0)
        N_nodes = E_idx.size(1)
        N_neighbors = E_idx.size(2)
        ii = torch.arange(N_nodes, device=device, dtype=torch.float32).view((1, -1, 1))
        d = (E_idx.float() - ii).unsqueeze(-1)
        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32, device=device)
            * -(np.log(10000.0) / self.num_embeddings)
        )
        # Grid-aligned
        # frequency = 2. * np.pi * torch.exp(
        #     -torch.linspace(
        #         np.log(self.period_range[0]), 
        #         np.log(self.period_range[1]),
        #         self.num_embeddings / 2
        #     )
        # )
        angles = d * frequency.view((1,1,1,-1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

class RNAFeatures(nn.Module):
    def __init__(self, edge_features, node_features, dist_map, num_positional_embeddings=16, num_node_embeddings=16,
                 num_rbf=16, top_k=30, block_size=3000, augment_eps=0., dropout=0.1):
        """ Extract protein features """
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.dist_map = dist_map
        self.top_k = top_k
        dists, idxs = torch.topk(self.dist_map, self.top_k, dim=-1, largest=False)
        self.dists = dists
        self.E_idx = idxs

        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        # Positional encoding
        self.edge_pos_embeddings = PositionalEncodings(num_positional_embeddings)
        self.node_pos_embeddings = nn.Embedding(block_size, num_node_embeddings)
        self.dropout = nn.Dropout(dropout)
        
        # Normalization and embedding
        node_in, edge_in = num_node_embeddings, (num_positional_embeddings + num_rbf)
        self.node_embedding = nn.Linear(node_in,  node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = Normalize(node_features)
        self.norm_edges = Normalize(edge_features)

    def _rbf(self, D):
        # Distance radial basis function
        device = D.device
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)

        # for i in range(D_count):
        #     fig = plt.figure(figsize=(4,4))
        #     ax = fig.add_subplot(111)
        #     rbf_i = RBF.data.numpy()[0,i,:,:]
        #     # rbf_i = D.data.numpy()[0,0,:,:]
        #     plt.imshow(rbf_i, aspect='equal')
        #     plt.axis('off')
        #     plt.tight_layout()
        #     plt.savefig('rbf{}.pdf'.format(i))
        #     print(np.min(rbf_i), np.max(rbf_i), np.mean(rbf_i))
        # exit(0)
        return RBF


    def forward(self, X):
        """ Featurize coordinates as an attributed graph """
        bs = X.size(0)
        device = X.device
        pos = torch.arange(X.size(1), device=X.device).repeat(X.size(0), 1)
        V = self.node_pos_embeddings(pos)


        E_idx = self.E_idx.unsqueeze(0).expand(bs, -1, -1).to(device)
        RBF = self._rbf(self.dists.to(device)).expand(bs, -1, -1, -1)
        E_positional = self.edge_pos_embeddings(E_idx)
        E = torch.cat((RBF, E_positional), -1).to(torch.float)

        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return V, E, E_idx

