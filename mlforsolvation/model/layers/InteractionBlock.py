'''
File to define custom GNN Layers
'''

import torch

import numpy as np

from torch_geometric.nn import MessagePassing



class InteractionBlock(MessagePassing):
    
    def __init__(self, in_channels, out_channels, hidden_channels, num_rbf, cond_mp=False, cutoff=0.7):
        super().__init__(aggr='add')
        
        self.register_buffer('_cutoff', torch.tensor(cutoff))
        self.register_buffer('_FREQUENCIES', torch.pi * torch.arange(1, num_rbf + 1))

        self.silu = torch.nn.SiLU()

        # Message MLP
        self.message1 = torch.nn.Linear(2 * in_channels + num_rbf, hidden_channels)
        self.message2 = torch.nn.Linear(hidden_channels, hidden_channels)
        # Node MLP
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

        self.cond_mp = cond_mp


    def forward(self, x, edge_index, edge_attributes, gamma_Q=1.0, beta_Q=0.0):

        # buildsinkernel_type: (R1: torch.Tensor)
        edge_attributes = self.buildsinkernel(R1=edge_attributes)

        # propagate_type: (x: torch.Tensor, edge_attributes: torch.Tensor)
        m = self.propagate(edge_index, x=x, edge_attributes=edge_attributes, size=None)
        if self.cond_mp:
            m = gamma_Q * m + beta_Q
        x = x + m

        # Nodewise combination
        x = self.lin1(x)
        x = self.silu(x)
        x = self.lin2(x)

        return x


    def message(self, x_i, x_j, edge_attributes):
        x = torch.concat((x_i, x_j, edge_attributes), dim=1)
        x = self.message1(x)
        x = self.silu(x)
        x = self.message2(x)
        x = self.silu(x)
        return x


    def buildsinkernel(self, R1):
        d_scaled = R1 * (1 / self._cutoff)
        # envelope_type: (R1: torch.Tensor)
        d_cutoff = self.envelope(R1=d_scaled)
        return d_cutoff * torch.sin(self._FREQUENCIES * d_scaled)


    def envelope(self,R1):
        p = 5 + 1
        a = -(p + 1) * (p + 2) / 2
        b = p * (p + 2)
        c = -p * (p + 1) / 2
        env_val = 1.0 / R1 + a * R1 ** (p - 1) + b * R1 ** p + c * R1 ** (p + 1)
        return env_val
