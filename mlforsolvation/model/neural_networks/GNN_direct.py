'''
File to define Neural Networks
'''

import torch

from torch import nn

from mlforsolvation.data.cosmo_radii import nm_radii
from mlforsolvation.model.layers.GNNGBNeck import GNN_GBNeck
from mlforsolvation.model.layers.GNNGrapher import GNN_Grapher
from mlforsolvation.model.layers.InteractionBlock import InteractionBlock
from mlforsolvation.model.layers.ElementEmbedding import ElementEmbedding
from mlforsolvation.model.layers.TotalChargeEmbedding import TotalChargeEmbedding
from mlforsolvation.model.layers.PartialChargeEmbedding import PartialChargeEmbedding



KJMOL_TO_KCALMOL = 0.239006



class ZQaEmbeddingCond(torch.nn.Module):

    def __init__(self, num_features, num_qbf):
        super().__init__()

        self.element_embedding = ElementEmbedding(num_features)
        self.partial_charge_embedding = PartialChargeEmbedding(num_features, num_qbf)

        self.gamma_Z = nn.Sequential(
            torch.nn.Linear(num_features, num_features // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(num_features // 2, num_features)
        )
        self.beta_Z = nn.Sequential(
            torch.nn.Linear(num_features, num_features // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(num_features // 2, num_features)
        )

        self.layer_norm = nn.LayerNorm(num_features)
    

    def forward(self, Z, Qa, batch_seg=None):
        ez = self.element_embedding(Z)
        eqa = self.partial_charge_embedding(Qa)

        mult_Z = self.gamma_Z(ez)
        add_Z = self.beta_Z(ez)

        normalized = self.layer_norm(eqa)

        h = mult_Z * normalized + add_Z

        return h, 1.0, 0.0



class ZQaQEmbeddingCond(torch.nn.Module):

    def __init__(self, num_features, num_qbf, Q_max=5, Q_min=0):
        super().__init__()

        self.element_embedding = ElementEmbedding(num_features)
        self.total_charge_embedding = TotalChargeEmbedding(num_features=num_features, Q_max=Q_max, Q_min=Q_min)
        self.partial_charge_embedding = PartialChargeEmbedding(num_features, num_qbf)

        self.gamma_Z = nn.Sequential(
            torch.nn.Linear(num_features, num_features // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(num_features // 2, num_features)
        )
        self.beta_Z = nn.Sequential(
            torch.nn.Linear(num_features, num_features // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(num_features // 2, num_features)
        )

        self.gamma_Q = nn.Sequential(
            torch.nn.Linear(num_features, num_features // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(num_features // 2, num_features)
        )
        self.beta_Q = nn.Sequential(
            torch.nn.Linear(num_features, num_features // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(num_features // 2, num_features)
        )

        self.layer_norm = nn.LayerNorm(num_features)
    

    def forward(self, Z, Qa, Q, batch_seg):
        ez = self.element_embedding(Z)
        eq = self.total_charge_embedding(Q)
        eqa = self.partial_charge_embedding(Qa)

        mult_Z = self.gamma_Z(ez)
        add_Z = self.beta_Z(ez)

        mult_Q = self.gamma_Q(eq)[batch_seg]
        add_Q = self.beta_Q(eq)[batch_seg]

        normalized = self.layer_norm(eqa)

        h = mult_Q * mult_Z * normalized + add_Z + add_Q

        return h, mult_Q, add_Q



class ZQaEmbeddingAdd(torch.nn.Module):

    def __init__(self, num_features, num_qbf):
        super().__init__()

        self.wr = torch.nn.Parameter(torch.tensor([1.0]))
        self.wz = torch.nn.Parameter(torch.tensor([1.0]))
        self.wqa = torch.nn.Parameter(torch.tensor([1.0]))

        self.element_embedding = ElementEmbedding(num_features)
        self.partial_charge_embedding = PartialChargeEmbedding(num_features, num_qbf)
    

    def forward(self, Z, Qa, batch_seg=None):
        ez = self.element_embedding(Z)
        eqa = self.partial_charge_embedding(Qa)
        return self.wz * ez + self.wqa * eqa, 1.0, 0.0


class ZQaQEmbeddingAdd(torch.nn.Module):

    def __init__(self, num_features, num_qbf, Q_max=5, Q_min=0):
        super().__init__()

        self.wr = torch.nn.Parameter(torch.tensor([1.0]))
        self.wz = torch.nn.Parameter(torch.tensor([1.0]))
        self.wq = torch.nn.Parameter(torch.tensor([1.0]))
        self.wqa = torch.nn.Parameter(torch.tensor([1.0]))

        self.element_embedding = ElementEmbedding(num_features)
        self.total_charge_embedding = TotalChargeEmbedding(num_features=num_features, Q_max=Q_max, Q_min=Q_min)
        self.partial_charge_embedding = PartialChargeEmbedding(num_features, num_qbf)
    

    def forward(self, Z, Qa, Q, batch_seg):
        ez = self.element_embedding(Z)
        eq = self.total_charge_embedding(Q)[batch_seg]
        eqa = self.partial_charge_embedding(Qa)
        return self.wz * ez + self.wq * eq + self.wqa * eqa, 1.0, 0.0



class ZQaEmbeddingSimple(torch.nn.Module):

    def __init__(self, num_features, Qa_mean, Qa_var):
        super().__init__()

        self.register_buffer("Qa_mean", torch.as_tensor(Qa_mean))
        self.register_buffer("Qa_var", torch.as_tensor(Qa_var))

        self.proj = nn.Linear(2, num_features)
    

    def forward(self, Z, Qa, batch_seg=None):
        Z_norm = (Z / 53.0).unsqueeze(1)
        Qa_norm = (Qa - self.Qa_mean) / self.Qa_var
        h = torch.concat((Z_norm, Qa_norm), dim=1)
        return self.proj(h), 1.0, 0.0
    

class ZQaQEmbeddingSimple(torch.nn.Module):

    def __init__(self, num_features, Qa_mean, Qa_var, Q_max=5):
        super().__init__()

        self.register_buffer("Qa_mean", torch.as_tensor(Qa_mean))
        self.register_buffer("Qa_var", torch.as_tensor(Qa_var))
        self.Q_max = Q_max if Q_max > 0 else 1.0

        self.proj = nn.Linear(3, num_features)
    

    def forward(self, Z, Qa, Q, batch_seg):
        Z_norm = (Z / 53.0).unsqueeze(1)
        Qa_norm = (Qa - self.Qa_mean) / self.Qa_var
        Q_norm = (Q / self.Q_max)[batch_seg].unsqueeze(1)
        h = torch.concat((Z_norm, Qa_norm, Q_norm), dim=1)
        return self.proj(h), 1.0, 0.0



class GNN_direct_heads(GNN_GBNeck, GNN_Grapher):

    def __init__(
        self,
        num_features,
        num_rbf,
        eps,
        num_qbf=None,
        embd_type="add",
        use_q_embd=False,
        cond_mp=False,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=100,
        Qa_mean=None,
        Qa_var=None,
        Q_max=5,
        Q_min=0,
    ):
        gbneck_radius = 10.0
        self._gnn_radius = radius
        GNN_GBNeck.__init__(self, eps=eps, radius=gbneck_radius, max_num_neighbors=max_num_neighbors)
        GNN_Grapher.__init__(self, radius=radius, max_num_neighbors=max_num_neighbors)

        self._fraction = fraction
        self.Q_min = Q_min
        self.Q_max = Q_max
        self.use_q_embd = use_q_embd

        self.embedding = self.build_embedding_layer(
            embd_type=embd_type,
            use_q_embd=use_q_embd,
            num_features=num_features,
            num_qbf=num_qbf,
            Qa_mean=Qa_mean,
            Qa_var=Qa_var
        )

        num_heads = Q_max - Q_min + 1
        if num_heads <= 0:
            raise ValueError(f"Invalid charge range: Q_min={Q_min}, Q_max={Q_max}")
        self.readout_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_features, num_features),
                nn.SiLU(),
                nn.Linear(num_features, 1)
            )
            for _ in range(num_heads)
        ])

        self.interaction1 = InteractionBlock(num_features, num_features, num_features, num_rbf, cutoff=radius, cond_mp=cond_mp)
        self.interaction2 = InteractionBlock(num_features, num_features, num_features, num_rbf, cutoff=radius, cond_mp=cond_mp)
        self.interaction3 = InteractionBlock(num_features, num_features, num_features, num_rbf, cutoff=radius, cond_mp=cond_mp)

        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

        self._nobatch = False
    

    def build_embedding_layer(
        self,
        embd_type,
        use_q_embd,
        num_features,
        num_qbf=None,
        Qa_mean=None,
        Qa_var=None
    ):
        if embd_type == "simple":
            assert Qa_mean is not None and Qa_var is not None
            if use_q_embd:
                return ZQaQEmbeddingSimple(num_features, Qa_mean, Qa_var, self.Q_max)
            else:
                return ZQaEmbeddingSimple(num_features, Qa_mean, Qa_var)
        
        elif embd_type == "add":
            assert num_qbf is not None
            if use_q_embd:
                return ZQaQEmbeddingAdd(num_features, num_qbf, self.Q_max, self.Q_min)
            else:
                return ZQaEmbeddingAdd(num_features, num_qbf)
        
        elif embd_type == "cond":
            assert num_qbf is not None
            if use_q_embd:
                return ZQaQEmbeddingCond(num_features, num_qbf, self.Q_max, self.Q_min)
            else:
                return ZQaEmbeddingCond(num_features, num_qbf)
        
        else:
            raise ValueError(f"Unknown embedding type: {embd_type}")
 


    def forward(self, data):
        data.pos = data.pos.clone().detach().requires_grad_(True)
        
        gnn_edge_index, gnn_edge_attributes = self.build_gnn_graph(data)

        Z = data.z
        Qa = data.qa_aimnet2.unsqueeze(1)

        if self.use_q_embd:
            Q = data.q
            gnn_in, gamma_Q, beta_Q = self.embedding(Z, Qa, Q, batch_seg=data.batch)
        else:
            gnn_in, gamma_Q, beta_Q = self.embedding(Z, Qa, batch_seg=data.batch)

        gnn_energies = self.interaction1(edge_index=gnn_edge_index, x=gnn_in, edge_attributes=gnn_edge_attributes, gamma_Q=gamma_Q, beta_Q=beta_Q)
        gnn_energies = self.silu(gnn_energies)
        gnn_energies = self.interaction2(edge_index=gnn_edge_index, x=gnn_energies, edge_attributes=gnn_edge_attributes, gamma_Q=gamma_Q, beta_Q=beta_Q)
        gnn_energies = self.silu(gnn_energies)
        gnn_energies = self.interaction3(edge_index=gnn_edge_index, x=gnn_energies, edge_attributes=gnn_edge_attributes, gamma_Q=gamma_Q, beta_Q=beta_Q)
        
        q_idx = torch.abs(data.q).to(torch.long).squeeze(-1) - self.Q_min
        q_idx = q_idx[data.batch]
        energies = torch.empty((gnn_energies.size(0), 1), dtype=gnn_energies.dtype, device=gnn_energies.device)
        for qi in range(self.Q_max - self.Q_min + 1):
            mask = q_idx == qi
            energies[mask] = self.readout_heads[qi](gnn_energies[mask])
        
        energy = torch.empty((torch.max(data.batch) + 1, 1), device=energies.device)
        for batch in data.batch.unique():
            energy[batch] = energies[torch.where(data.batch == batch)].sum()

        return energy.squeeze(1) * KJMOL_TO_KCALMOL



class GNN_direct(GNN_GBNeck, GNN_Grapher):

    def __init__(
        self,
        num_features,
        num_rbf,
        eps,
        num_qbf=None,
        embd_type="add",
        use_q_embd=False,
        cond_mp=False,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=32,
        Qa_mean=None,
        Qa_var=None,
        Q_max=5,
        Q_min=0,
    ):
        gbneck_radius = 10.0
        self._gnn_radius = radius
        GNN_GBNeck.__init__(self, eps=eps, radius=gbneck_radius, max_num_neighbors=max_num_neighbors)
        GNN_Grapher.__init__(self, radius=radius, max_num_neighbors=max_num_neighbors)

        self.embedding = self.build_embedding_layer(
            embd_type=embd_type,
            use_q_embd=use_q_embd,
            num_features=num_features,
            num_qbf=num_qbf,
            Qa_mean=Qa_mean,
            Qa_var=Qa_var,
            Q_max=Q_max,
            Q_min=Q_min
        )

        self._fraction = fraction
        self._nobatch = False
        self.use_q_embd = use_q_embd

        self.interaction1 = InteractionBlock(num_features, num_features, num_features, num_rbf, cutoff=radius, cond_mp=cond_mp)
        self.interaction2 = InteractionBlock(num_features, num_features, num_features, num_rbf, cutoff=radius, cond_mp=cond_mp)
        self.interaction3 = InteractionBlock(num_features, 1, num_features, num_rbf, cutoff=radius, cond_mp=cond_mp)

        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
    

    def build_embedding_layer(
        self,
        embd_type,
        use_q_embd,
        num_features,
        num_qbf=None,
        Qa_mean=None,
        Qa_var=None,
        Q_max=5,
        Q_min=0
    ):
        if embd_type == "simple":
            assert Qa_mean is not None and Qa_var is not None
            if use_q_embd:
                return ZQaQEmbeddingSimple(num_features, Qa_mean, Qa_var, Q_max)
            else:
                return ZQaEmbeddingSimple(num_features, Qa_mean, Qa_var)
        
        elif embd_type == "add":
            assert num_qbf is not None
            if use_q_embd:
                return ZQaQEmbeddingAdd(num_features, num_qbf, Q_max, Q_min)
            else:
                return ZQaEmbeddingAdd(num_features, num_qbf)
        
        elif embd_type == "cond":
            assert num_qbf is not None
            if use_q_embd:
                return ZQaQEmbeddingCond(num_features, num_qbf, Q_max, Q_min)
            else:
                return ZQaEmbeddingCond(num_features, num_qbf)
        
        else:
            raise ValueError(f"Unknown embedding type: {embd_type}")


    def forward(self, data):
        data.pos = data.pos.clone().detach().requires_grad_(True)
    
        gnn_edge_index, gnn_edge_attributes = self.build_gnn_graph(data)

        Z = data.z
        Qa = data.qa_aimnet2.unsqueeze(1)

        if self.use_q_embd:
            Q = data.q
            gnn_in, gamma_Q, beta_Q = self.embedding(Z, Qa, Q, batch_seg=data.batch)
        else:
            gnn_in, gamma_Q, beta_Q = self.embedding(Z, Qa, batch_seg=data.batch)

        energies = self.interaction1(edge_index=gnn_edge_index, x=gnn_in, edge_attributes=gnn_edge_attributes, gamma_Q=gamma_Q, beta_Q=beta_Q)
        energies = self.silu(energies)
        energies = self.interaction2(edge_index=gnn_edge_index, x=energies, edge_attributes=gnn_edge_attributes, gamma_Q=gamma_Q, beta_Q=beta_Q)
        energies = self.silu(energies)
        energies = self.interaction3(edge_index=gnn_edge_index, x=energies, edge_attributes=gnn_edge_attributes, gamma_Q=gamma_Q, beta_Q=beta_Q)
        
        energy = torch.empty((torch.max(data.batch) + 1, 1), device=energies.device)
        for batch in data.batch.unique():
            energy[batch] = energies[torch.where(data.batch == batch)].sum()

        return energy.squeeze(1) * KJMOL_TO_KCALMOL
    
