import torch



class TotalChargeEmbedding(torch.nn.Module):

    def __init__(self, num_features, Q_max=5, Q_min=0):
        super().__init__()

        self.Q_min = Q_min

        self.embd = torch.nn.Embedding(Q_max - Q_min + 1, num_features)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_features),
            torch.nn.SiLU(),
            torch.nn.Linear(num_features, num_features)
        )


    def forward(self, Q):
        return self.mlp(self.embd(torch.abs(Q) - self.Q_min))