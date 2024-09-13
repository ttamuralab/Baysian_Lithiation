import numpy as np
import os

from Lib import CG
from Lib import struct_out_to_poscar
from Lib import poscar


def main(ifname):
    base = poscar.poscar(f"target_structure/{ifname}")
    base_h_11 = base.h[1][1]
    ib = 0
    nLi = base.natm[-1]
    work_dir = "Li200"

    if not os.path.isfile(f"target_structure/POSCAR_Li{nLi}_CG"):
        CG.main(f"target_structure/{ifname}", f"{work_dir}/Li{nLi}/CG")
        struct_out_to_poscar.main(
            f"Li200/Li{nLi}/CG/base_and_mol.STRUCT_OUT",
            f"target_structure/POSCAR_Li{nLi}_CG",
        )

    CG_base = f"target_structure/POSCAR_Li{nLi}_CG"
    if not os.path.isfile(f"target_structure/POSCAR_Li{nLi}_stretched"):
        while True:
            ib += 1
            nb = 1 + (ib) * 2 / 100
            print(nb)
            base.h[1][1] = base_h_11 * nb

            CG.main(
                CG_base,
                f"{work_dir}/Li{nLi}/stretch/{nb}",
                nb,
            )

            os.makedirs(
                f"Li200/Li{nLi}/stretch/poscar",
                exist_ok=True,
            )
            struct_out_to_poscar.main(
                f"Li200/Li{nLi}/stretch/{nb}/base_and_mol.STRUCT_OUT",
                f"Li200/Li{nLi}/stretch/poscar/POSCAR_Li{nLi}_{nb}",
            )
            CG_base = f"Li200/Li{nLi}/stretch/poscar/POSCAR_Li{nLi}_{nb}"

            fname_log = f"Li200/Li{nLi}/stretch/{nb}/log_SIESTA"
            with open(fname_log) as f:
                for i, line in reversed(list(enumerate(f))):
                    if "Voigt" in line:
                        break
            print(nLi, nb, float(line.split()[-5]))
            if float(line.split()[-5]) > 0:
                struct_out_to_poscar.main(
                    f"Li200/Li{nLi}/stretch/{nb}/base_and_mol.STRUCT_OUT",
                    f"target_structure/POSCAR_Li{nLi}_stretched",
                )
                break


def isbond(base):
    isbond = []
    combi = []
    for i in range(99):
        for j in range(i + 1, 100, 1):
            # print(i, j)
            norm = base.ra[i] - base.ra[j] - np.round(base.ra[i] - base.ra[j])
            norm = np.dot(base.h, norm)
            norm = np.linalg.norm(norm)
            # if base.catm[base.katm[i]] == "Si" and base.catm[base.katm[j]] == "Si":
            #     if norm < 2.6:
            #         isbond.append(True)
            #     else:
            #         isbond.append(False)
            #     combi.append(f"{i+1}, {j+1}")
            if base.catm[base.katm[i]] == "Si" and base.catm[base.katm[j]] == "O":
                if norm < 1.8:
                    isbond.append(True)
                else:
                    isbond.append(False)
                combi.append(f"{i+1}, {j+1}")

            # if i == 17 and j == 58:
            #     print(norm)
            #     print(isbond[-1])
            # elif base.catm[base.katm[i]] == "O" and base.catm[base.katm[j]] == "Si":
            #     if norm < 1.9:
            #         isbond.append(True)
            #     else:
            #         isbond.append(False)
    # print(isbond[-1])
    return isbond, combi


if __name__ == "__main__":
    fname_log = "Li189/Li10/0/SIESTA/0/log_SIESTA"
    with open(fname_log) as f:
        # for i, line in reversed(list(enumerate(f))):
        for i, line in enumerate(f):
            if "Voigt" in line:
                break
    print(float(line.split()[-5]))
