import torch



class COSMORadiiEmbedding(torch.nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1, num_features),
            torch.nn.SiLU(),
            torch.nn.Linear(num_features, num_features)
        )


    def forward(self, R_cosmo):
        return self.mlp(R_cosmo)