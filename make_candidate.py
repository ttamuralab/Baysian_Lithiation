import numpy as np
import time
import os
import json

from Lib import poscar


with open("settings.json", "r") as f:
    settings = json.load(f)
work_dir = settings["work_dir"]
fname_base = settings["fname_base"]
print("program make_candidate.py")
print("work_dir =", work_dir)
print("fname_base =", fname_base)

base = poscar.poscar(f"target_structure/{fname_base}")

xdiv = int(base.h[0][0] / 0.2)
ydiv = int(base.h[1][1] / 0.2)
zdiv = int(base.h[2][2] / 0.2)

start = time.time()
n_candidate = 0

# loop in xyz
for ix in range(xdiv):
    for iy in range(ydiv):
        for iz in range(zdiv):
            rx = ix / xdiv
            ry = iy / ydiv
            rz = iz / zdiv
            tx = base.h[0][0] * ix / xdiv
            ty = base.h[1][1] * iy / ydiv
            tz = base.h[2][2] * iz / zdiv
            coo_r = np.array([rx, ry, rz])
            coo_t = np.array([tx, ty, tz])

            # judge by distance
            min_norm = 100
            for ib in range(base.natm_all):
                d = coo_r - base.ra[ib] - np.round(coo_r - base.ra[ib])
                d = np.dot(base.h, d)
                norm = np.linalg.norm(d)
                if norm < min_norm:
                    min_norm = norm

            if 1.8 < min_norm:
                try:
                    candidate_r = np.block([[candidate_r], [np.array([ix, iy, iz])]])
                    candidate_t = np.block([[candidate_t], [coo_t]])
                except:
                    candidate_r = np.array([ix, iy, iz])
                    candidate_t = coo_t

                n_candidate += 1
                print(n_candidate, candidate_r[-1])

print("elapsed_time to create candidate =", time.time() - start)

os.makedirs(f"./{work_dir}/candidate", exist_ok=True)
candidate_r = candidate_r[np.lexsort(np.fliplr(candidate_r).T)]
np.savetxt(
    f"./{work_dir}/candidate/candidate_cooang_r_all.dat",
    candidate_r,
    fmt="%i",
)
candidate_t = candidate_t[np.lexsort(np.fliplr(candidate_t).T)]
np.savetxt(
    f"./{work_dir}/candidate/candidate_cooang_t_all.dat",
    candidate_t,
    fmt=["%f", "%f", "%f"],
)