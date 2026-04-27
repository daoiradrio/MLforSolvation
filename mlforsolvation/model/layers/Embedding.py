import torch

import torch.nn as nn



def build_embedding_layer(embd_type, num_features=None, Qa_mean=None, Qa_stfd=None):
    if embd_type == "simple":
        assert Qa_mean is not None and Qa_stfd is not None, "Qa_mean and Qa_stfd must be provided for simple embedding"
        return ZQaEmbeddingSimple(Qa_mean, Qa_stfd)
    elif embd_type == "add":
        return ZQaEmbeddingAdd()
    elif embd_type == "cond":
        return ZQaEmbeddingCond(num_features)
    else:
        raise ValueError(f"Unknown embedding type: {embd_type}")



class ZQaQEmbeddingCond(torch.nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.gamma_Z = nn.Sequential(
            torch.nn.Linear(num_features, num_features),
            torch.nn.SiLU(),
            torch.nn.Linear(num_features, num_features)
        )
        self.beta_Z = nn.Sequential(
            torch.nn.Linear(num_features, num_features),
            torch.nn.SiLU(),
            torch.nn.Linear(num_features, num_features)
        )

        self.gamma_Q = nn.Sequential(
            torch.nn.Linear(num_features, num_features),
            torch.nn.SiLU(),
            torch.nn.Linear(num_features, num_features)
        )
        self.beta_Q = nn.Sequential(
            torch.nn.Linear(num_features, num_features),
            torch.nn.SiLU(),
            torch.nn.Linear(num_features, num_features)
        )

        self.layer_norm = nn.LayerNorm(num_features)
    

    def forward(self, ez, eq, eqa, batch_seg=None):
        mult_Z = self.gamma_Z(ez)
        add_Z = self.beta_Z(ez)

        mult_Q = self.gamma_Q(eq)
        add_Q = self.beta_Q(eq)

        normalized = self.layer_norm(eqa)

        h0 = mult_Z * normalized + add_Z
        h = mult_Q[batch_seg] * h0 + add_Q[batch_seg]

        return h



class ZQaEmbeddingCond(torch.nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.gamma_Z = nn.Sequential(
            torch.nn.Linear(num_features, num_features),
            torch.nn.SiLU(),
            torch.nn.Linear(num_features, num_features)
        )
        self.beta_Z = nn.Sequential(
            torch.nn.Linear(num_features, num_features),
            torch.nn.SiLU(),
            torch.nn.Linear(num_features, num_features)
        )

        self.layer_norm = nn.LayerNorm(num_features)
    

    def forward(self, ez, eqa):
        mult_Z = self.gamma_Z(ez)
        add_Z = self.beta_Z(ez)

        normalized = self.layer_norm(eqa)

        h = mult_Z * normalized + add_Z

        return h



class ZQaQEmbeddingAdd(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.wz = torch.nn.Parameter(torch.tensor([1.0]))
        self.wq = torch.nn.Parameter(torch.tensor([1.0]))
        self.wqa = torch.nn.Parameter(torch.tensor([1.0]))
    

    def forward(self, ez, eq, eqa):
        return self.wz * ez + self.wq * eq + self.wqa * eqa



class ZQaEmbeddingAdd(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.wz = torch.nn.Parameter(torch.tensor([1.0]))
        self.wqa = torch.nn.Parameter(torch.tensor([1.0]))
    

    def forward(self, ez, eqa):
        return self.wz * ez + self.wqa * eqa



class ZQaEmbeddingSimple(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.sigma_z = 1.0
        self.sigma_qa = 1.0
    

    def forward(self, ez, eqa):
        return torch.concat(((1 / self.sigma_z**2) * ez, (1 / self.sigma_qa**2) * eqa))


        Bc = nm_radii[data.z].unsqueeze(1)
        Z = data.z#.unsqueeze(1)
        Qa = data.qa_aimnet2.unsqueeze(1)
        #Bcn = torch.concat((Bc, Z.unsqueeze(1), Qa), dim=1)



class ZQaQEmbeddingSimple(torch.nn.Module):

    def __init__(self):
        super().__init__()
    

    def forward(self, ez, eq, eqa):
        return ez, eq, eqa
