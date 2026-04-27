import torch



bohr_radii = torch.full((54,), torch.nan)
ang_radii  = torch.full((54,), torch.nan)

# H  (1)
bohr_radii[1] = 2.4944
ang_radii[1]  = 1.3200

# B  (5)
bohr_radii[5] = 4.3539
ang_radii[5]  = 2.3040

# C  (6)
bohr_radii[6] = 3.8550
ang_radii[6]  = 2.0400

# N  (7)
bohr_radii[7] = 3.5149
ang_radii[7]  = 1.8600

# O  (8)
bohr_radii[8] = 3.4469
ang_radii[8]  = 1.8240

# F  (9)
bohr_radii[9] = 3.3335
ang_radii[9]  = 1.7640

# Si (14)
bohr_radii[14] = 4.7621
ang_radii[14]  = 2.5200

# P  (15)
bohr_radii[15] = 4.0818
ang_radii[15]  = 2.1600

# S  (16)
bohr_radii[16] = 4.0818
ang_radii[16]  = 2.1600

# Cl (17)
bohr_radii[17] = 3.9684
ang_radii[17]  = 2.1000

# As (33)
bohr_radii[33] = 4.1952
ang_radii[33]  = 2.2200

# Se (34)
bohr_radii[34] = 4.3086
ang_radii[34]  = 2.2800

# Br (35)
bohr_radii[35] = 4.1952
ang_radii[35]  = 2.2200

# I  (53)
bohr_radii[53] = 4.4900
ang_radii[53]  = 2.3760



nm_radii = ang_radii * 0.1
