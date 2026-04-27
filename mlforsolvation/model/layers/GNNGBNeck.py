import torch

import torch.nn as nn

from torch_geometric.transforms import RadiusGraph
from torch.nn import PairwiseDistance

from mlforsolvation.data.cosmo_radii import nm_radii
from mlforsolvation.model.layers.GBNeck import GBNeck



class GNN_GBNeck(nn.Module):
    def __init__(self, eps, radius=0.4, max_num_neighbors=32):
        '''
        GNN to reproduce the GBNeck Model
        '''
        super().__init__()

        # Initiate Graph Builder
        self._radius = radius
        self._max_num_neighbors = max_num_neighbors
        self._grapher = RadiusGraph(r=self._radius, loop=False, max_num_neighbors=self._max_num_neighbors)

        # Init Distance Calculation
        self._distancer = PairwiseDistance()
        self.calculate_energy = GBNeck(eps)

        self.lin = nn.Linear(1,1)


    def build_graph(self, data):

        # Get Radius Graph
        graph = self._grapher(data)

        # Extract edge index
        edge_index = graph.edge_index

        # Extract edge features
        distances = self._distancer(data.pos[edge_index[0]], data.pos[edge_index[1]])

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)

        return edge_index, edge_attributes


    def forward(self, data):

        # Enable tracking of gradients
        # Get input as Tensor create on device
        data.pos = data.pos.clone().detach().requires_grad_(True)
        # Build Graph
        edge_index, edge_attributes = self.build_graph(data)

        # Do message passing
        Bc = nm_radii[data.z]
        energies = self.calculate_energy(x=Bc, edge_index=edge_index, edge_attributes=edge_attributes)

        # Return prediction and Gradients with respect to data
        gradients = torch.autograd.grad(energies.sum(), inputs=data.pos, create_graph=True)[0]
        forces = -1 * gradients

        if self._nobatch:
            energy = energies.sum()
            energy = energy.unsqueeze(0)
            energy = energy.unsqueeze(0)
        else:
            energy = torch.empty((torch.max(data.batch) + 1,1), device=self._device)
            for batch in data.batch.unique():
                energy[batch] = energies[torch.where(data.batch == batch)].sum()

        return energy, forces
