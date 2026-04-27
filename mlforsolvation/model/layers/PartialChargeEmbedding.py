import torch
import os

import numpy as np



class PartialChargeEmbedding(torch.nn.Module):

    def __init__(self, num_features, num_qbf, alpha=0.85):
        super().__init__()

        #compute training dataset statistics of partial charges (quantiles, scaled difference between quantiles)
        partial_charges = np.loadtxt(os.path.join(os.getcwd(), "partial_charges.dat"))
        ps = (np.arange(num_qbf) + 0.5) / num_qbf
        mus = np.quantile(partial_charges, ps)
        sigmas = [alpha * (mus[1] - mus[0])]
        for i in range(1, num_qbf-1):
            sigmas.append(alpha * (mus[i+1] - mus[i-1]) / 2)
        sigmas.append(alpha * (mus[-1] - mus[-2]))

        #initialize RBF embedding buffers
        mus = torch.from_numpy(mus)
        sigmas = torch.tensor(sigmas)

        self.register_buffer("mus", mus)
        self.register_buffer("sigmas", sigmas)

        #initialize projection layer after RBF embedding
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(num_qbf, num_features//2),
            torch.nn.SiLU(),
            torch.nn.Linear(num_features//2, num_features)
        )
    

    def forward(self, q):
        #q = q.unsqueeze(-1)
        rbf = torch.exp(-0.5 * ((q - self.mus) / self.sigmas) ** 2)
        return self.mlp(rbf)
