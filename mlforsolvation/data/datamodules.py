import torch

import numpy as np

from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from random import shuffle



N_MAX = 350

EV_TO_HARTREE = 0.0367492929
HARTREE_TO_EV = 27.2114079527
HARTREE_TO_KCALMOL = 627.50961
KCALMOL_TO_EV = 0.04336
EV_TO_KCALMOL = 23.060542

ANGSTROM_TO_BOHR = 1.8897259886
BOHR_TO_ANGSTROM = 1 / ANGSTROM_TO_BOHR



class MLQSolvDataset(Dataset):

    def __init__(self, data_path, split_set, partial_charges, solvation, cutoff=5.0, num_samples=None, Q_max=5, Q_min=0):
        super().__init__()

        self.cutoff = cutoff
        self.num_atoms_max = np.loadtxt(f"{data_path}/num_atoms_max.dat", dtype=int)
        num_data = np.loadtxt(f"{data_path}/{split_set}/num_data.dat", dtype=int)

        self.IDX = np.loadtxt(f"{data_path}/{split_set}/IDX.dat", dtype=str)

        N_memmap = np.memmap(f"{data_path}/{split_set}/N.npy", dtype="int32", mode="r", shape=(num_data,))
        Z_memmap = np.memmap(f"{data_path}/{split_set}/Z.npy", dtype="int32", mode="r", shape=(num_data, self.num_atoms_max))
        R_memmap = np.memmap(f"{data_path}/{split_set}/R.npy", dtype="float32", mode="r", shape=(num_data, self.num_atoms_max, 3)) * BOHR_TO_ANGSTROM
        Q_memmap = np.memmap(f"{data_path}/{split_set}/Q.npy", dtype="int32", mode="r", shape=(num_data,))
        Qa_aimnet2_memmap = np.memmap(f"{data_path}/{split_set}/Qa.npy", dtype="float32", mode="r", shape=(num_data, self.num_atoms_max))
        M_memmap = np.memmap(f"{data_path}/{split_set}/M.npy", dtype="bool", mode="r", shape=(num_data, self.num_atoms_max))
        E_aimnet2_memmap = np.memmap(f"{data_path}/{split_set}/E_aimnet2.npy", dtype="float32", mode="r", shape=(num_data,))
        E_memmap = np.memmap(f"{data_path}/{split_set}/E.npy", dtype="float32", mode="r", shape=(num_data,))

        if partial_charges == "mulliken":
            Qa_memmap = np.memmap(f"{data_path}/{split_set}/MULLIKEN.npy", dtype="float32", mode="r", shape=(num_data, self.num_atoms_max))
        elif partial_charges == "loewdin":
            Qa_memmap = np.memmap(f"{data_path}/{split_set}/LOEWDIN.npy", dtype="float32", mode="r", shape=(num_data, self.num_atoms_max))
        elif partial_charges == "hirshfeld":
            Qa_memmap = np.memmap(f"{data_path}/{split_set}/HIRSHFELD.npy", dtype="float32", mode="r", shape=(num_data, self.num_atoms_max))
        
        if solvation == "cpcm":
            E_solv_memmap = np.memmap(f"{data_path}/{split_set}/E_solv_cpcm.npy", dtype="float32", mode="r", shape=(num_data,)) * HARTREE_TO_KCALMOL
        elif solvation == "cosmors":
            E_solv_memmap = np.memmap(f"{data_path}/{split_set}/E_solv_cosmors.npy", dtype="float32", mode="r", shape=(num_data,))
        
        Q_mask = (Q_min <= np.abs(Q_memmap)) * (np.abs(Q_memmap) <= Q_max)
        self.num_samples = min(num_samples, Q_mask.sum()) if num_samples is not None else Q_mask.sum()
        sample_idx = [i for i in range(self.num_samples)]
        shuffle(sample_idx)
        sample_idx = np.array(sample_idx)

        self.N = torch.from_numpy(np.asarray(N_memmap)[Q_mask][sample_idx].copy())
        self.Z = torch.from_numpy(np.asarray(Z_memmap)[Q_mask][sample_idx].copy())
        self.R = torch.from_numpy(np.asarray(R_memmap)[Q_mask][sample_idx].copy())
        self.Q = torch.from_numpy(np.asarray(Q_memmap)[Q_mask][sample_idx].copy())
        self.Qa = torch.from_numpy(np.asarray(Qa_memmap)[Q_mask][sample_idx].copy())
        self.Qa_aimnet2 = torch.from_numpy(np.asarray(Qa_aimnet2_memmap)[Q_mask][sample_idx].copy())
        self.M = torch.from_numpy(np.asarray(M_memmap)[Q_mask][sample_idx].copy())
        self.E = torch.from_numpy(np.asarray(E_memmap)[Q_mask][sample_idx].copy())
        self.E_solv = torch.from_numpy(np.asarray(E_solv_memmap)[Q_mask][sample_idx].copy())
        self.E_aimnet2 = torch.from_numpy(np.asarray(E_aimnet2_memmap)[Q_mask][sample_idx].copy())

        del N_memmap
        del Z_memmap
        del R_memmap
        del Q_memmap
        del Qa_memmap
        del Qa_aimnet2_memmap
        del M_memmap
        del E_memmap
        del E_solv_memmap
        del E_aimnet2_memmap


    def __len__(self):
        return self.num_samples


    def __getitem__(self, i):

        E = torch.tensor([self.E[i]])
        E_solv = torch.tensor([self.E_solv[i]])

        N = torch.tensor([self.N[i]])
        Z = self.Z[i, :]
        R = self.R[i, :, :]
        Q = torch.tensor([self.Q[i]])
        Qa = self.Qa[i, :]
        Qa_aimnet2 = self.Qa_aimnet2[i, :]
        M = self.M[i, :]

        Z = Z[M]
        R = R[M, :]
        Qa = Qa[M]
        Qa_aimnet2 = Qa_aimnet2[M]

        data = Data(
            n=N.to(torch.long),
            pos=R.to(torch.float32),
            z=Z.to(torch.long),

            q=Q.to(torch.long),
            qa=Qa.to(torch.float32),
            qa_aimnet2=Qa_aimnet2.to(torch.float32),
            e=E.to(torch.float32),
            e_solv=E_solv.to(torch.float32),
        )

        return data
    


class MLQSolvDatamodule(torch.nn.Module):

    def __init__(self, num_train, num_val, num_test, partial_charges, solvation, data_path, cutoff, Q_max, Q_min):
        super().__init__()
        self.train_dataset = MLQSolvDataset(
            data_path=data_path,
            split_set="train",
            num_samples=num_train,
            partial_charges=partial_charges,
            solvation=solvation,
            cutoff=cutoff,
            Q_max=Q_max,
            Q_min=Q_min
        )
        self.val_dataset = MLQSolvDataset(
            data_path=data_path,
            split_set="val",
            num_samples=num_val,
            partial_charges=partial_charges,
            solvation=solvation,
            cutoff=cutoff,
            Q_max=Q_max,
            Q_min=Q_min
        )
        self.test_dataset = MLQSolvDataset(
            data_path=data_path,
            split_set="test",
            num_samples=num_test,
            partial_charges=partial_charges,
            solvation=solvation,
            cutoff=cutoff,
            Q_max=Q_max,
            Q_min=Q_min
        )


    def collate_fn(self, batch):
        return Batch.from_data_list(batch)
