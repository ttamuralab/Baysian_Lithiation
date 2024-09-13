import os
import numpy as np
import json
import random
import time

from Lib.candidate import Candidate
from Lib.cal_energy import cal_energy
from Lib import poscar
from Lib.BOLib.BayesianOptimization import BayesianOptimization


def get_delete_index(vec, rcut):
    for i in range(vec.shape[0]):
        # *相対座標化
        vec[i] = np.dot(base.h_inv, vec[i])
        # *周期境界条件
        vec[i] = vec[i] - np.round(vec[i])
        # *絶対座標化
        vec[i] = np.dot(base.h, vec[i])
    norm = np.linalg.norm(vec, axis=1)
    delete_index = np.where(norm < rcut)

    return delete_index


def process_of_end_program(work_dir, candidate, fname_base, fname_mol):
    energy_searched = candidate.y_current
    # insert most stable site
    coo = candidate.cooang_t_searched_current[np.argmin(energy_searched)]

    print("added site =", coo)
    print("added site energy =", np.min(energy_searched))

    base = poscar.poscar(f"target_structure/{fname_base}")
    mol = poscar.poscar(f"target_structure/{fname_mol}")
    base.add_mol(mol, coo)
    nLi = base.natm[base.catm.index("Li")]
    base.write(f"target_structure/POSCAR_Li{nLi}")

    rcut = 1.8

    # delete most stable site from candidate
    delete_index = get_delete_index(candidate.cooang_t_searched_current - coo, rcut)
    candidate.cooang_t_searched_current = np.delete(
        candidate.cooang_t_searched_current, delete_index, 0
    )
    candidate.y_current = np.delete(candidate.y_current, delete_index)

    # delete near site
    vec = candidate.cooang_t_unsearched - coo

    for i in range(vec.shape[0]):
        vec[i] = np.dot(base.h_inv, vec[i])
        vec[i] = vec[i] - np.round(vec[i])
        vec[i] = np.dot(base.h, vec[i])
    norm = np.linalg.norm(vec, axis=1)
    delete_index = np.where(norm < rcut)

    candidate.cooang_t_unsearched = np.delete(
        candidate.cooang_t_unsearched, delete_index, axis=0
    )
    candidate.cooang_r_unsearched = np.delete(
        candidate.cooang_r_unsearched, delete_index, axis=0
    )

    # add gauss correction to mean
    sig = 1.3
    gauss = 15 * np.exp(-((norm) ** 2) / sig**2)
    candidate.m0_c = candidate.m0_c + gauss
    candidate.m0_c = np.delete(candidate.m0_c, delete_index, axis=0)

    # calculate soap
    from ase import Atoms
    from ase.io import read, write
    from dscribe.descriptors import SOAP

    species = ["Si", "O", "Li"]
    soap = SOAP(species=species, periodic=True, r_cut=6, n_max=10, l_max=12, sigma=0.2)

    atoms = read(f"target_structure/POSCAR_Li{nLi}")
    Li = Atoms("Li", positions=[[0, 0, 0]])
    atoms.extend(Li)
    soap_Li = soap.create(atoms, centers=[-1])

    candidate.descriptor_unsearched = np.empty(
        [candidate.cooang_t_unsearched.shape[0], soap_Li.shape[1]]
    )
    start = time.time()
    for i in range(candidate.cooang_t_unsearched.shape[0]):
        atoms = read(f"target_structure/POSCAR_Li{nLi}")
        Li = Atoms("Li", positions=[])
        atoms.extend(Li)
        soap_Li = soap.create(atoms, centers=[-1])
        candidate.descriptor_unsearched[i] = soap_Li
    print("etime.soap =", time.time() - start)

    # PCA
    print("pca")
    from sklearn.decomposition import PCA

    start = time.time()

    pca = PCA()
    pca.fit(
        np.block([[candidate.descriptor_searched], [candidate.descriptor_unsearched]])
    )

    candidate.feature_searched = pca.transform(candidate.descriptor_searched)
    candidate.feature_searched = candidate.feature_searched[:, :25]

    candidate.feature_unsearched = pca.transform(candidate.descriptor_unsearched)
    candidate.feature_unsearched = candidate.feature_unsearched[:, :25]
    print("etime.soap =", time.time() - start)

    print("write")
    if nLi % 10 != 0:
        work_dir_next = f"{os.path.split(work_dir)[0]}/{nLi%10}"
        candidate.write(f"{work_dir_next}/candidate")


def process_of_end_loop():
    pass


random.seed(42)

# *__main__

if not os.path.isfile("settings.json"):
    print("settings.json is not exist")
    exit()
else:
    with open("settings.json", "r") as f:
        settings = json.load(f)
    print("read setting file")

work_dir = settings["work_dir"]
print(f"work_dir = {work_dir}")

nsearch_max_rand = settings["nsearch_max_rand"]
fname_base = settings["fname_base"]
fname_mol = settings["fname_mol"]

base = poscar.poscar(f"target_structure/{fname_base}")
mol = poscar.poscar(f"target_structure/{fname_mol}")

# read status file
if not os.path.isfile(f"./{work_dir}/status.json"):
    print("status.json が存在しません")
    status = {"nsearch": 0}
    nsearch = 0
else:
    with open(f"{work_dir}/status.json", "r") as f:
        status = json.load(f)
    nsearch = status["nsearch"]

# read candidate
candidate = Candidate()
candidate.read(f"{work_dir}/candidate")


# *cal_ads_energy
method_cal_energy = "SIESTA"
cal_energy = cal_energy(work_dir, method_cal_energy, fname_base, fname_mol)
if "energy_ads" in status:
    energy_ads = status["energy_ads"]
else:
    energy_base, energy_mol = cal_energy.ads(cutoff=100, PAO="DZP")
    energy_ads = energy_base - 14.719009

    status["energy_ads"] = energy_ads
    with open(f"./{work_dir}/status.json", "w") as f:
        json.dump(status, f, indent=2)


# *探索設定
option = ["rbf"] * candidate.feature_unsearched.shape[1]
bounds = [(0, None)] * candidate.feature_unsearched.shape[1]
if not "hparams" in status:
    beta = [0.00001] * candidate.feature_unsearched.shape[1]
else:
    beta = np.array(status["hparams"])
    if beta.shape[0] != candidate.feature_unsearched.shape[1]:
        beta = [0.00001] * candidate.feature_unsearched.shape[1]

beta = np.array(beta)

BO = BayesianOptimization(option, beta, bounds)
try:
    nLi = base.natm[base.catm.index("Li")]
except:
    nLi = 0

target_nLi = settings["target_nLi"]
# *探索
while True:
    if_rand_search = True
    try:
        if candidate.cooang_r_searched.ndim == 2:
            if candidate.cooang_r_searched.shape[0] >= 5:
                if_rand_search = False
    except:
        pass

    if if_rand_search:  # *ランダムサーチ
        print("random search")
        next_index = random.randrange(candidate.cooang_r_unsearched.shape[0])

    else:  # *BO
        print("BO")

        # *学習データ
        feature = candidate.feature_searched
        target = candidate.y

        # *探索
        start = time.time()

        if not candidate.m0_c is None and not candidate.m0_y is None:
            m0 = [candidate.m0_c, candidate.m0_y]
        else:
            m0 = None

        print(candidate.feature_unsearched.shape)
        next_index, _ = BO.one_step_run(
            feature,
            target,
            candidate.feature_unsearched,
            nsearch % 5 == 0,  # candidate.feature_searched.shape[0] % 5 == 0,
            m0=m0,
            min_y=np.min(candidate.y_current),
        )

        # exit()

        status["max_acq ="] = _["max_acq"]
        status["hparams"] = _["hparams"].tolist()

        if _["max_acq"] == 0:
            next_index = random.randrange(candidate.cooang_r_unsearched.shape[0])

        if _["max_acq"] < 0.027 and np.count_nonzero(_["hparams"] >= 10**-8) > 10:
            with open(f"./{work_dir}/status.json", "w") as f:
                json.dump(status, f, indent=2)
            process_of_end_program(work_dir, candidate, fname_base, fname_mol)
            # *statusの更新
            break

    # calculate energy
    energy = cal_energy.main(
        nsearch, candidate.cooang_t_unsearched[next_index], cutoff=100, PAO="DZP"
    )
    energy = energy - energy_ads

    candidate.update(next_index, energy)
    candidate.write(f"{work_dir}/candidate")

    nsearch = nsearch + 1
    # update status
    try:
        status["min_y"] = np.min(candidate.y_current)
    except:
        pass

    status["nsearch"] = nsearch
    status["n_candidate_searched"] = candidate.cooang_r_searched.shape[0]

    with open(f"./{work_dir}/status.json", "w") as f:
        json.dump(status, f, indent=2)

print("finish")
