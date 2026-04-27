#!/usr/bin/env python3
"""
Predict energy/forces with AIMNet2 and run geometry optimization via ASE.

Example:
  python predict_optimize_aimnet2.py \
    --xyz molecule.xyz \
    --total-charge 0 \
    --optimize \
    --optimizer lbfgs \
    --fmax 0.02 \
    --steps 300 \
    --traj opt.traj \
    --out-xyz opt.xyz

Notes:
- Atomic charges are inferred internally by AIMNet2.
- AIMNet2 returns energy in eV and forces in eV/Å (assumed).
"""

import argparse
from pathlib import Path

import numpy as np
import torch

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


def _to_torch(x, device, dtype=None):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


class AIMNet2ASECalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, aimnet_calc, aimnet_device, total_charge, **kwargs):
        super().__init__(**kwargs)
        self.aimnet = aimnet_calc
        self.aimnet_device = aimnet_device
        self.total_charge = int(total_charge)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        coord = torch.tensor(atoms.positions, dtype=torch.float32, device=self.aimnet_device)
        numbers = torch.tensor(atoms.numbers, dtype=torch.long, device=self.aimnet_device)
        charge = torch.tensor([self.total_charge], dtype=torch.long, device=self.aimnet_device)

        aimnet_inp = {"coord": coord, "numbers": numbers, "charge": charge}
        aimnet_out = self.aimnet(aimnet_inp, forces=True)

        energy = _to_torch(aimnet_out["energy"], device="cpu", dtype=torch.float32).sum()
        forces = _to_torch(aimnet_out["forces"], device="cpu", dtype=torch.float32)

        self.results["energy"] = float(energy.detach().cpu().item())
        self.results["forces"] = forces.detach().cpu().numpy()


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
    parser.add_argument("--xyz", required=True, type=Path, help="Input XYZ (or any ASE-readable format)")
    parser.add_argument("--total-charge", type=int, default=None, help="Total molecular charge")
    parser.add_argument("--aimnet-device", type=str, default=None, help="Device for AIMNet2 (defaults to CUDA if available)")

    parser.add_argument("--optimize", default=True, action="store_true", help="Run geometry optimization")
    default_opt = "preconlbfgs" if _HAVE_PRECON else "lbfgs"
    parser.add_argument("--optimizer", type=str, default=default_opt, help="fire|lbfgs|bfgs|bfgsls|preconlbfgs")
    parser.add_argument("--fmax", type=float, default=0.02, help="Convergence criterion (eV/Å)")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--traj", type=Path, default=None, help="Trajectory output (.traj)")
    parser.add_argument("--logfile", type=Path, default=None, help="Optimizer log file")
    parser.add_argument("--out-xyz", type=Path, default=None, help="Output XYZ for optimized structure")

    args = parser.parse_args()

    if not _HAVE_AIMNET:
        raise RuntimeError("aimnet is not installed. Install aimnet to use this script.")

    atoms = ase_io.read(args.xyz)

    if args.total_charge is None:
        total_charge = atoms.info.get("charge", 0)
    else:
        total_charge = args.total_charge

    aimnet_device = args.aimnet_device
    if aimnet_device is None:
        aimnet_device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        aimnet_calc = AIMNet2Calculator("aimnet2", device=str(aimnet_device))
    except TypeError:
        aimnet_calc = AIMNet2Calculator("aimnet2")

    atoms.calc = AIMNet2ASECalculator(
        aimnet_calc=aimnet_calc,
        aimnet_device=aimnet_device,
        total_charge=total_charge,
    )

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    print(f"Predicted energy (vacuum): {energy:.6f} eV")
    print(f"Max |force|: {np.linalg.norm(forces, axis=1).max():.6f} eV/Å")

    if args.optimize:
        opt = pick_optimizer(args.optimizer, atoms, args.traj, args.logfile)
        opt.run(fmax=args.fmax, steps=args.steps)

        if args.out_xyz is not None:
            ase_io.write(args.out_xyz, atoms)
            print(f"Wrote optimized structure to {args.out_xyz}")


if __name__ == "__main__":
    main()
