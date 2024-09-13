import numpy as np
from dscribe.descriptors import SOAP
import json
from ase import Atoms
from ase.io import read, write
import time


with open("settings.json", "r") as f:
    settings = json.load(f)
work_dir = settings["work_dir"]
nLi = settings["target_nLi"]
fname_base = settings["fname_base"]

print("program make_soap.py")
print("work_dir =", work_dir)

species = ["Si", "O", "Li"]
soap = SOAP(species=species, periodic=True, r_cut=6, n_max=10, l_max=12, sigma=0.2)

atoms = read(f"target_structure/{fname_base}")
Li = Atoms("Li", positions=[[0, 0, 0]])
atoms.extend(Li)
soap_Li = soap.create(atoms, centers=[-1])

# read
candidate_cooang_t_unsearched = np.loadtxt(
    f"{work_dir}/candidate/candidate_cooang_t_all.dat"
)

candidate_descriptor_unsearched = np.empty(
    [candidate_cooang_t_unsearched.shape[0], soap_Li.shape[1]]
)
# main
start = time.time()
for i in range(candidate_cooang_t_unsearched.shape[0]):
    atoms = read(f"target_structure/{fname_base}")
    Li = Atoms("Li", positions=[candidate_cooang_t_unsearched[i]])
    atoms.extend(Li)
    soap_Li = soap.create(atoms, centers=[-1])
    candidate_descriptor_unsearched[i] = soap_Li
print("etime.soap =", time.time() - start)

# save
np.savetxt(
    f"{work_dir}/candidate/candidate_descriptor_all.dat",
    candidate_descriptor_unsearched,
)
