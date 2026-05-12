import torch



class ElementEmbedding(torch.nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.embedding = torch.nn.Embedding(84, num_features)
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_features),
            torch.nn.SiLU(),
            torch.nn.Linear(num_features, num_features)
        )


    def forward(self, x):
        return self.projection(self.embedding(x))
