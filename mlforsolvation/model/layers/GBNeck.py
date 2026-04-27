import torch

from torch_geometric.nn import MessagePassing



class GBNeck(MessagePassing):

    def __init__(self, eps):
        '''
        GBNeck interaction
        '''
        super().__init__(aggr='add')
        self.register_buffer('soluteDielectric', torch.tensor(1.0))
        self.register_buffer('solventDielectric', torch.tensor(eps))


    def forward(self, edge_index, x, edge_attributes):

        # x = torch.concat((x.unsqueeze(1),charges.unsqueeze(1)),dim=1)
        pair_energies = self.propagate(edge_index, x=x, edge_attributes=edge_attributes,size=None)

        # do not doublecount
        pair_energies = pair_energies / 2

        # Nodewise combination
        single_energy = self.nodewise(x)

        return pair_energies + single_energy


    def nodewise(self, x):
        B = x[:,0]
        charge = x[:,1]

        energy = -0.5*138.935485*(1/self.soluteDielectric-1/self.solventDielectric)*charge**2/B

        return energy.unsqueeze(1)
    

    def message(self, x_i : torch.Tensor, x_j : torch.Tensor , edge_attributes : torch.Tensor) -> torch.Tensor:
        r = edge_attributes[:,0]
        B1 = x_i[:,0]
        B2 = x_j[:,0]
        charge1 = x_i[:,1]
        charge2 = x_j[:,1]

        f = torch.sqrt(r**2 + B1*B2*torch.exp(-r**2/(4*B1*B2)))
        energy = -138.935485*(1/self.soluteDielectric-1/self.solventDielectric)*charge1*charge2/f

        return energy.unsqueeze(1)
