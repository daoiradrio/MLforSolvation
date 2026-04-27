#!/usr/bin/env python3
"""
Predict energy/forces with a trained GNN_star and run geometry optimization via ASE.

Example:
  python predict_optimize_gnn_star.py \
    --checkpoint /path/to/best_model.ckpt \
    --config /path/to/config.txt \
    --xyz molecule.xyz \
    --total-charge 0 \
    --optimize \
    --optimizer preconlbfgs \
    --fmax 0.02 \
    --steps 300 \
    --traj opt.traj \
    --out-xyz opt.xyz

Notes:
- Atomic partial charges are inferred on-the-fly by AIMNet2 and passed to GNN_star.
- GNN_star energies are in kcal/mol and are converted to eV before summation with AIMNet2 vacuum energy.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from torch_geometric.data import Data

from ase import io as ase_io
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS, BFGSLineSearch, FIRE, LBFGS

try:
    from ase.optimize.precon import PreconLBFGS, Exp
    _HAVE_PRECON = True
except Exception:
    _HAVE_PRECON = False

try:
    from aimnet.calculators import AIMNet2Calculator
    _HAVE_AIMNET = True
except Exception:
    _HAVE_AIMNET = False

KCALMOL_TO_EV = 0.0433641153  # consistent with ase.units.kcal/mol



eps = {
    "water": 80.0,
    "acetonitrile": 36.0,
    "cyclohexane": 2.0
}



def _str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def parse_config(path: Path) -> dict:
    cfg = {}
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("--"):
                key, val = line[2:].split("=", 1)
                cfg[key.strip()] = val.strip()
    return cfg


def load_qa_stats(args, cfg):
    if args.qa_mean is not None and args.qa_var is not None:
        return float(args.qa_mean), float(args.qa_var)

    if args.qa_stats is not None:
        vals = np.loadtxt(args.qa_stats, dtype=float).reshape(-1)
        if vals.size == 2:
            return float(vals[0]), float(vals[1])
        raise ValueError("--qa-stats must contain exactly two numbers: mean and variance")

    data_path = args.data_path or cfg.get("data_path", None)
    partial_charges = args.partial_charges or cfg.get("partial_charges", None)
    solvation = args.solvation or cfg.get("solvation", None)
    cutoff = float(cfg.get("cutoff", 5.0))
    q_max = int(cfg.get("max_total_charge", 0))
    q_min = int(cfg.get("min_total_charge", 0))

    if data_path is None or partial_charges is None or solvation is None:
        raise ValueError(
            "Qa_mean/Qa_var not provided. Supply --qa-mean/--qa-var, --qa-stats, "
            "or pass --data-path/--partial-charges/--solvation to recompute from training data."
        )

    from mlqsolv_riniker.data.datamodules import MLQSolvDataset

    ds = MLQSolvDataset(
        data_path=data_path,
        split_set="train",
        partial_charges=partial_charges,
        solvation=solvation,
        cutoff=cutoff,
        Q_max=q_max,
        Q_min=q_min,
    )
    qa = ds.Qa[ds.M]
    qa_mean = qa.mean().item()
    qa_var = (qa.std().item()) ** 2

    np.savetxt(
        fname="partial_charges.dat",
        X=ds.Qa[ds.M].flatten().tolist(),
        fmt="%.6f"
    )

    return qa_mean, qa_var


def validate_total_charge(total_charge, q_min, q_max):
    if total_charge < q_min or total_charge > q_max:
        raise ValueError(
            f"Total charge {total_charge} outside training range [{q_min}, {q_max}]."
        )
    return total_charge


def _to_torch(x, device, dtype=None):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


class GNNStarCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model, aimnet_calc, device, aimnet_device, total_charge, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.aimnet = aimnet_calc
        self.device = device
        self.aimnet_device = aimnet_device
        self.total_charge = int(total_charge)

        q_tensor = torch.tensor([self.total_charge], dtype=torch.long, device=device)
        self._q_tensor = q_tensor

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        n_atoms = len(atoms)

        coord = torch.tensor(atoms.positions, dtype=torch.float32, device=self.aimnet_device)
        numbers = torch.tensor(atoms.numbers, dtype=torch.long, device=self.aimnet_device)
        total_charge = torch.tensor([self.total_charge], dtype=torch.long, device=self.aimnet_device)

        aimnet_inp = {"coord": coord, "numbers": numbers, "charge": total_charge}
        aimnet_out = self.aimnet(aimnet_inp, forces=True)
        energy_vac = _to_torch(aimnet_out["energy"], device=self.device, dtype=torch.float32).sum()
        forces_vac = _to_torch(aimnet_out["forces"], device=self.device, dtype=torch.float32)
        qa = _to_torch(aimnet_out["charges"], device=self.device, dtype=torch.float32).reshape(-1)

        pos = torch.tensor(atoms.positions, dtype=torch.float32, device=self.device)
        z = torch.tensor(atoms.numbers, dtype=torch.long, device=self.device)
        batch = torch.zeros(n_atoms, dtype=torch.long, device=self.device)

        data = Data(
            pos=pos,
            z=z,
            qa=qa,
            qa_aimnet2=qa,
            q=self._q_tensor,
            n=torch.tensor([n_atoms], dtype=torch.long, device=self.device),
            batch=batch,
        )

        with torch.enable_grad():
            energy_kcal, _ = self.model(data)
            energy_kcal = energy_kcal.sum()
            forces_kcal = -torch.autograd.grad(energy_kcal, data.pos, create_graph=False)[0]

        energy_solv_ev = energy_kcal * KCALMOL_TO_EV
        forces_solv_ev = forces_kcal * KCALMOL_TO_EV

        energy_total = energy_vac + energy_solv_ev
        forces_total = forces_vac + forces_solv_ev

        self.results["energy"] = float(energy_total.detach().cpu().item())
        self.results["forces"] = forces_total.detach().cpu().numpy()
        self.results["energy_vac"] = float(energy_vac.detach().cpu().item())
        self.results["energy_solv"] = float(energy_solv_ev.detach().cpu().item())



def build_model(cfg, solvent, qa_mean, qa_var, checkpoint_path, device):
    use_q_heads = _str2bool(cfg.get("use_q_heads", "False"))
    use_q_embd = _str2bool(cfg.get("use_q_embd", "False"))
    cond_mp = _str2bool(cfg.get("cond_mp", "False"))

    num_features = int(cfg["num_features"])
    num_rbf = int(cfg["num_rbf"])
    num_qbf = int(cfg["num_qbf"])
    embd_type = cfg["embd_type"]
    cutoff = float(cfg.get("cutoff", 5.0))
    q_max = int(cfg.get("max_total_charge", 0))
    q_min = int(cfg.get("min_total_charge", 0))

    if use_q_heads:
        from mlqsolv_riniker.model.networks.GNN_star import GNN_star_heads as Model
    else:
        from mlqsolv_riniker.model.networks.GNN_star import GNN_star as Model

    model = Model(
        num_features=num_features,
        num_rbf=num_rbf,
        num_qbf=num_qbf,
        fraction=0.5,
        eps=eps[solvent],
        radius=cutoff,
        max_num_neighbors=100,
        embd_type=embd_type,
        use_q_embd=use_q_embd,
        cond_mp=cond_mp,
        Qa_mean=qa_mean,
        Qa_var=qa_var,
        Q_max=q_max,
        Q_min=q_min,
    ).to(dtype=torch.float32, device=device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model


def pick_optimizer(name, atoms, traj, logfile):
    name = name.lower()
    if name == "fire":
        return FIRE(atoms, trajectory=traj, logfile=logfile, maxstep=0.2)
    if name == "lbfgs":
        return LBFGS(atoms, trajectory=traj, logfile=logfile)
    if name == "bfgs":
        return BFGS(atoms, trajectory=traj, logfile=logfile)
    if name == "bfgsls":
        return BFGSLineSearch(atoms, trajectory=traj, logfile=logfile)
    if name == "preconlbfgs":
        if not _HAVE_PRECON:
            raise RuntimeError("ASE preconditioner not available. Install ase>=3.22.")
        precon = Exp(A=3.0, r_cut=3.0)
        return PreconLBFGS(atoms, precon=precon, trajectory=traj, logfile=logfile)
    raise ValueError(f"Unknown optimizer '{name}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to model checkpoint (*.ckpt)")
    parser.add_argument("--config", required=True, type=Path, help="Path to training config.txt")

    parser.add_argument("--xyz", required=True, type=Path, help="Input XYZ (or any ASE-readable format)")
    parser.add_argument("--total-charge", type=int, default=None, help="Total molecular charge")

    parser.add_argument("--qa-mean", type=float, default=None)
    parser.add_argument("--qa-var", type=float, default=None)
    parser.add_argument("--qa-stats", type=Path, default=None, help="Text file with: <mean> <variance>")
    parser.add_argument("--data-path", type=Path, default=None, help="Training data path (optional; to recompute Qa stats)")
    parser.add_argument("--partial-charges", type=str, default=None, help="Training partial charge model name")
    parser.add_argument("--solvation", type=str, default=None, help="Training solvation model name")
    parser.add_argument("--solvent", type=str, default=None, help="Training solvent")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--aimnet-device", type=str, default=None, help="Device for AIMNet2 (defaults to --device)")

    parser.add_argument("--optimize", default=True, action="store_true", help="Run geometry optimization")
    default_opt = "preconlbfgs" if _HAVE_PRECON else "lbfgs"
    parser.add_argument("--optimizer", type=str, default=default_opt, help="fire|lbfgs|bfgs|bfgsls|preconlbfgs")
    parser.add_argument("--fmax", type=float, default=0.02, help="Convergence criterion (eV/Å)")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--traj", type=Path, default=None, help="Trajectory output (.traj)")
    parser.add_argument("--logfile", type=Path, default=None, help="Optimizer log file")
    parser.add_argument("--out-xyz", type=Path, default=None, help="Output XYZ for optimized structure")

    args = parser.parse_args()

    cfg = parse_config(args.config)
    qa_mean, qa_var = load_qa_stats(args, cfg)

    atoms = ase_io.read(args.xyz)

    #q_max = int(cfg.get("max_total_charge", 5))
    #q_min = int(cfg.get("min_total_charge", 0))
    q_max = 5
    q_min = 0
    if args.total_charge is None:
        total_charge = atoms.info.get("charge", 0)
    else:
        total_charge = args.total_charge
    total_charge = validate_total_charge(int(total_charge), q_min, q_max)

    device = torch.device(args.device)
    aimnet_device = torch.device(args.aimnet_device) if args.aimnet_device else device
    model = build_model(cfg, args.solvent, qa_mean, qa_var, args.checkpoint, device)

    if not _HAVE_AIMNET:
        raise RuntimeError("aimnet is not installed. Please install aimnet to use this script.")
    try:
        aimnet_calc = AIMNet2Calculator("aimnet2", device=str(aimnet_device))
    except TypeError:
        aimnet_calc = AIMNet2Calculator("aimnet2")

    atoms.calc = GNNStarCalculator(
        model=model,
        aimnet_calc=aimnet_calc,
        device=device,
        aimnet_device=aimnet_device,
        total_charge=total_charge,
    )

    energy = atoms.get_potential_energy()  # triggers model
    forces = atoms.get_forces()
    if "energy_vac" in atoms.calc.results and "energy_solv" in atoms.calc.results:
        e_vac = atoms.calc.results["energy_vac"]
        e_solv = atoms.calc.results["energy_solv"]
        print(f"Predicted energy (vacuum): {e_vac:.6f} eV")
        print(f"Predicted energy (solv):   {e_solv:.6f} eV")
    print(f"Predicted energy (total): {energy:.6f} eV")
    print(f"Max |force|: {np.linalg.norm(forces, axis=1).max():.6f} eV/Å")

    if args.optimize:
        opt = pick_optimizer(args.optimizer, atoms, args.traj, args.logfile)
        opt.run(fmax=args.fmax, steps=args.steps)

        if args.out_xyz is not None:
            ase_io.write(args.out_xyz, atoms)
            print(f"Wrote optimized structure to {args.out_xyz}")


if __name__ == "__main__":
    main()
