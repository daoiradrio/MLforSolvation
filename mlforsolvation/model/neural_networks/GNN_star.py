'''
File to define Neural Networks
'''

import torch

from torch import nn

from mlforsolvation.data.cosmo_radii import nm_radii
from mlforsolvation.model.layers.GNNGBNeck import GNN_GBNeck
from mlforsolvation.model.layers.GNNGrapher import GNN_Grapher
from mlforsolvation.model.layers.InteractionBlock import InteractionBlock
from mlforsolvation.model.layers.COSMORadiiEmbedding import COSMORadiiEmbedding
from mlforsolvation.model.layers.ElementEmbedding import ElementEmbedding
from mlforsolvation.model.layers.TotalChargeEmbedding import TotalChargeEmbedding
from mlforsolvation.model.layers.PartialChargeEmbedding import PartialChargeEmbedding



KJMOL_TO_KCALMOL = 0.239006



class RZQaEmbeddingCond(torch.nn.Module):

    def __init__(self, num_features, num_qbf):
        super().__init__()

        self.cosmo_radii_embedding = COSMORadiiEmbedding(num_features)
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

        self.proj = torch.nn.Linear(2 * num_features, num_features)
    

    def forward(self, R_cosmo, Z, Qa, batch_seg=None):
        ez = self.element_embedding(Z)
        eqa = self.partial_charge_embedding(Qa)
        er = self.cosmo_radii_embedding(R_cosmo)
        mult_Z = self.gamma_Z(ez)
        add_Z = self.beta_Z(ez)

        normalized = self.layer_norm(eqa)

        h = mult_Z * normalized + add_Z
        h = torch.concat((er, h), dim=1)
        h = self.proj(h)

        return h, 1.0, 0.0



class RZQaQEmbeddingCond(torch.nn.Module):

    def __init__(self, num_features, num_qbf, Q_max=5, Q_min=0):
        super().__init__()

        self.cosmo_radii_embedding = COSMORadiiEmbedding(num_features)
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
        
        self.proj = torch.nn.Linear(2 * num_features, num_features)
    

    def forward(self, R_cosmo, Z, Qa, Q, batch_seg):
        er = self.cosmo_radii_embedding(R_cosmo)
        ez = self.element_embedding(Z)
        eq = self.total_charge_embedding(Q)
        eqa = self.partial_charge_embedding(Qa)

        mult_Z = self.gamma_Z(ez)
        add_Z = self.beta_Z(ez)

        mult_Q = self.gamma_Q(eq)[batch_seg]
        add_Q = self.beta_Q(eq)[batch_seg]

        normalized = self.layer_norm(eqa)

        h = mult_Q * mult_Z * normalized + add_Z + add_Q
        h = torch.concat((er, h), dim=1)
        h = self.proj(h)

        return h, mult_Q, add_Q



class RZQaEmbeddingAdd(torch.nn.Module):

    def __init__(self, num_features, num_qbf):
        super().__init__()

        nan_mask = ~torch.isnan(nm_radii)
        r_min = torch.min(nm_radii[nan_mask])
        r_max = torch.max(nm_radii[nan_mask])
        self.register_buffer("r_min", r_min)
        self.register_buffer("r_max", r_max)

        self.wr = torch.nn.Parameter(torch.tensor([1.0]))
        self.wz = torch.nn.Parameter(torch.tensor([1.0]))
        self.wqa = torch.nn.Parameter(torch.tensor([1.0]))

        self.cosmo_radii_embedding = COSMORadiiEmbedding(num_features)
        self.element_embedding = ElementEmbedding(num_features)
        self.partial_charge_embedding = PartialChargeEmbedding(num_features, num_qbf)

        self.proj = torch.nn.Linear(2 * num_features, num_features)
    

    def forward(self, R_cosmo, Z, Qa, batch_seg=None):
        er = self.cosmo_radii_embedding(R_cosmo)
        ez = self.element_embedding(Z)
        eqa = self.partial_charge_embedding(Qa)
        h = self.wz * ez + self.wqa * eqa
        h = torch.concat((er, h), dim=1)
        h = self.proj(h)
        return h, 1.0, 0.0


class RZQaQEmbeddingAdd(torch.nn.Module):

    def __init__(self, num_features, num_qbf, Q_max=5, Q_min=0):
        super().__init__()

        nan_mask = ~torch.isnan(nm_radii)
        r_min = torch.min(nm_radii[nan_mask])
        r_max = torch.max(nm_radii[nan_mask])
        self.register_buffer("r_min", r_min)
        self.register_buffer("r_max", r_max)

        self.wr = torch.nn.Parameter(torch.tensor([1.0]))
        self.wz = torch.nn.Parameter(torch.tensor([1.0]))
        self.wq = torch.nn.Parameter(torch.tensor([1.0]))
        self.wqa = torch.nn.Parameter(torch.tensor([1.0]))

        self.cosmo_radii_embedding = COSMORadiiEmbedding(num_features)
        self.element_embedding = ElementEmbedding(num_features)
        self.total_charge_embedding = TotalChargeEmbedding(num_features=num_features, Q_max=Q_max, Q_min=Q_min)
        self.partial_charge_embedding = PartialChargeEmbedding(num_features, num_qbf)

        self.proj = torch.nn.Linear(2 * num_features, num_features)
    

    def forward(self, R_cosmo, Z, Qa, Q, batch_seg):
        er = self.cosmo_radii_embedding(R_cosmo)
        ez = self.element_embedding(Z)
        eq = self.total_charge_embedding(Q)[batch_seg]
        eqa = self.partial_charge_embedding(Qa)
        h = self.wq * eq + self.wqa * eqa + self.wz * ez
        h = torch.concat((er, h), dim=1)
        h = self.proj(h)
        return h, 1.0, 0.0



class RZQaEmbeddingSimple(torch.nn.Module):

    def __init__(self, num_features, Qa_mean, Qa_var):
        super().__init__()

        nan_mask = ~torch.isnan(nm_radii)
        r_min = torch.min(nm_radii[nan_mask])
        r_max = torch.max(nm_radii[nan_mask])
        self.register_buffer("r_min", r_min)
        self.register_buffer("r_max", r_max)

        self.register_buffer("Qa_mean", torch.as_tensor(Qa_mean))
        self.register_buffer("Qa_var", torch.as_tensor(Qa_var))

        self.proj = torch.nn.Linear(3, num_features)
    

    def forward(self, R_cosmo, Z, Qa, batch_seg=None):
        R_norm = (R_cosmo - self.r_min) / (self.r_max - self.r_min)
        Z_norm = (Z / 53.0).unsqueeze(1)
        Qa_norm = (Qa - self.Qa_mean) / self.Qa_var
        h = torch.concat((R_norm, Z_norm, Qa_norm), dim=1)
        return self.proj(h), 1.0, 0.0
    


class RZQaQEmbeddingSimple(torch.nn.Module):

    def __init__(self, num_features, Qa_mean, Qa_var, Q_max=5):
        super().__init__()

        nan_mask = ~torch.isnan(nm_radii)
        r_min = torch.min(nm_radii[nan_mask])
        r_max = torch.max(nm_radii[nan_mask])
        self.register_buffer("r_min", r_min)
        self.register_buffer("r_max", r_max)

        self.register_buffer("Qa_mean", torch.as_tensor(Qa_mean))
        self.register_buffer("Qa_var", torch.as_tensor(Qa_var))
        self.Q_max = Q_max if Q_max > 0 else 1.0

        self.proj = torch.nn.Linear(4, num_features)
    

    def forward(self, R_cosmo, Z, Qa, Q, batch_seg):
        R_norm = (R_cosmo - self.r_min) / (self.r_max - self.r_min)
        Z_norm = (Z / 53.0).unsqueeze(1)
        Qa_norm = (Qa - self.Qa_mean) / self.Qa_var
        Q_norm = (Q / self.Q_max)[batch_seg].unsqueeze(1)
        h = torch.concat((R_norm, Z_norm, Qa_norm, Q_norm), dim=1)
        return self.proj(h), 1.0, 0.0



class GNN_star_heads(GNN_GBNeck, GNN_Grapher):

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
        self.register_buffer("nm_radii", nm_radii)

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
                nn.Linear(num_features, 2)
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
                return RZQaQEmbeddingSimple(num_features, Qa_mean, Qa_var, self.Q_max)
            else:
                return RZQaEmbeddingSimple(num_features, Qa_mean, Qa_var)
        
        elif embd_type == "add":
            assert num_qbf is not None
            if use_q_embd:
                return RZQaQEmbeddingAdd(num_features, num_qbf, Q_max=self.Q_max, Q_min=self.Q_min)
            else:
                return RZQaEmbeddingAdd(num_features, num_qbf)
        
        elif embd_type == "cond":
            assert num_qbf is not None
            if use_q_embd:
                return RZQaQEmbeddingCond(num_features, num_qbf, Q_max=self.Q_max, Q_min=self.Q_min)
            else:
                return RZQaEmbeddingCond(num_features, num_qbf)
        
        else:
            raise ValueError(f"Unknown embedding type: {embd_type}")
 


    def forward(self, data):
        data.pos = data.pos.clone().detach().requires_grad_(True)

        edge_index, edge_attributes = self.build_graph(data)
        gnn_edge_index, gnn_edge_attributes = self.build_gnn_graph(data)

        Bc = self.nm_radii[data.z].unsqueeze(1)
        Z = data.z
        Qa = data.qa_aimnet2.unsqueeze(1)
        if self.use_q_embd:
            Q = data.q
            Bcn, gamma_Q, beta_Q = self.embedding(Bc, Z, Qa, Q, batch_seg=data.batch)
        else:
            Bcn, gamma_Q, beta_Q = self.embedding(Bc, Z, Qa, batch_seg=data.batch)
        
        Bcn = self.interaction1(edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes, gamma_Q=gamma_Q, beta_Q=beta_Q)
        Bcn = self.silu(Bcn)
        Bcn = self.interaction2(edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes, gamma_Q=gamma_Q, beta_Q=beta_Q)
        Bcn = self.silu(Bcn)
        Bcn = self.interaction3(edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes, gamma_Q=gamma_Q, beta_Q=beta_Q)

        q_idx = torch.abs(data.q).to(torch.long).squeeze(-1) - self.Q_min
        q_idx = q_idx[data.batch]
        Bcn_out = torch.empty((Bcn.size(0), 2), dtype=Bcn.dtype, device=Bcn.device)
        for qi in range(self.Q_max - self.Q_min + 1):
            mask = q_idx == qi
            Bcn_out[mask] = self.readout_heads[qi](Bcn[mask])
        
        Bcn_out[:, 0] = Bc[:, 0] * (self._fraction + self.sigmoid(Bcn_out[:, 0])*(1-self._fraction)*2)

        energies = self.calculate_energy(x=Bcn_out, edge_index=edge_index, edge_attributes=edge_attributes)

        energy = torch.empty((torch.max(data.batch) + 1, 1), device=energies.device)
        for batch in data.batch.unique():
            energy[batch] = energies[torch.where(data.batch == batch)].sum()
        
        return energy.squeeze(1) * KJMOL_TO_KCALMOL, Bcn[:, 1]



class GNN_star(GNN_GBNeck, GNN_Grapher):

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
        self.register_buffer("nm_radii", nm_radii)

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
        self.interaction3 = InteractionBlock(num_features, 2, num_features, num_rbf, cutoff=radius, cond_mp=cond_mp)

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
                return RZQaQEmbeddingSimple(num_features, Qa_mean, Qa_var, Q_max)
            else:
                return RZQaEmbeddingSimple(num_features, Qa_mean, Qa_var)
        
        elif embd_type == "add":
            assert num_qbf is not None
            if use_q_embd:
                return RZQaQEmbeddingAdd(num_features, num_qbf, Q_max=Q_max, Q_min=Q_min)
            else:
                return RZQaEmbeddingAdd(num_features, num_qbf)
        
        elif embd_type == "cond":
            assert num_qbf is not None
            if use_q_embd:
                return RZQaQEmbeddingCond(num_features, num_qbf, Q_max=Q_max, Q_min=Q_min)
            else:
                return RZQaEmbeddingCond(num_features, num_qbf)
        
        else:
            raise ValueError(f"Unknown embedding type: {embd_type}")


    def forward(self, data):
        data.pos = data.pos.clone().detach().requires_grad_(True)
        # Build Graph
        edge_index, edge_attributes = self.build_graph(data)
        gnn_edge_index, gnn_edge_attributes = self.build_gnn_graph(data)

        # Do message passing
        Bc = self.nm_radii[data.z].unsqueeze(1)
        Z = data.z
        Qa = data.qa_aimnet2.unsqueeze(1)
        if self.use_q_embd:
            Q = data.q
            Bcn, gamma_Q, beta_Q = self.embedding(Bc, Z, Qa, Q, batch_seg=data.batch)
        else:
            Bcn, gamma_Q, beta_Q = self.embedding(Bc, Z, Qa, batch_seg=data.batch)

        Bcn = self.interaction1(edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes, gamma_Q=gamma_Q, beta_Q=beta_Q)
        Bcn = self.silu(Bcn)
        Bcn = self.interaction2(edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes, gamma_Q=gamma_Q, beta_Q=beta_Q)
        Bcn = self.silu(Bcn)
        Bcn = self.interaction3(edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes, gamma_Q=gamma_Q, beta_Q=beta_Q)
        
        Bcn[:, 0] = Bc[:, 0] * (self._fraction + self.sigmoid(Bcn[:, 0])*(1-self._fraction)*2)

        energies = self.calculate_energy(x=Bcn, edge_index=edge_index, edge_attributes=edge_attributes)
    
        energy = torch.empty((torch.max(data.batch) + 1, 1), device=energies.device)
        for batch in data.batch.unique():
            energy[batch] = energies[torch.where(data.batch == batch)].sum()
        
        return energy.squeeze(1) * KJMOL_TO_KCALMOL, Bcn[:, 1]



class dummy_GNN_star(GNN_GBNeck, GNN_Grapher):

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
        self.register_buffer("nm_radii", nm_radii)


    def forward(self, data):
        data.pos = data.pos.clone().detach().requires_grad_(True)

        edge_index, edge_attributes = self.build_graph(data)

        r = self.nm_radii[data.z].unsqueeze(1)
        Qa = data.qa_aimnet2.unsqueeze(1)
        Bcn = torch.concat((r, Qa), dim=1)
        energies = self.calculate_energy(x=Bcn, edge_index=edge_index, edge_attributes=edge_attributes)

        energy = torch.empty((torch.max(data.batch) + 1, 1), device=energies.device)
        for batch in data.batch.unique():
            energy[batch] = energies[torch.where(data.batch == batch)].sum()
        
        return energy.squeeze(1) * KJMOL_TO_KCALMOL, Bcn[:, 1]
