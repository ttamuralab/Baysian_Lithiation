import numpy as np
import subprocess
import os

from Lib import poscar
from Lib import poscar_to_fdf


class cal_energy:
    def __init__(
        self,
        work_dir,
        method,
        fname_base,
        fname_mol,
        cutoff=300,
        PAO=None,
        is_plus_U=False,
    ):
        self.work_dir = work_dir
        self.method = method
        self.fname_base = fname_base
        self.fname_mol = fname_mol
        self.cutoff = (cutoff,)
        self.PAO = PAO
        self.is_plus_U = is_plus_U

    def SIESTA(
        self,
        poscar,
        fname_fdf,
        savepath,
        ncore=24,
    ):
        work_dir = self.work_dir

        # copy base_dir to current
        command = f"cp -r {work_dir}/SIESTA/base/* ."
        subprocess.run(command, shell=True)
        poscar_to_fdf.main(
            poscar,
            f"./{fname_fdf}",
            cutoff=self.cutoff,
            PAO=self.PAO,
            is_plus_U=self.is_plus_U,
        )

        # run SIESTA
        cmd = f"mpirun -np {ncore} siesta -fdf XML.Write {fname_fdf} > log_SIESTA"
        subprocess.run(cmd, shell=True)

        # get energy from log
        fname_log = "log_SIESTA"
        with open(fname_log) as f:
            for i, line in enumerate(f):
                if "Total =" in line:
                    break

        os.makedirs(
            savepath,
            exist_ok=True,
        )
        # move storage dir
        fnames = [
            "*.psf",
            "*.ion",
            "*.yml",
            "*.fdf",
            "*.xml",
            "log_*",
            "BASIS_*",
            "PARALLEL_DIST",
            "NON_TRIMMED_KP_LIST",
            "CLOCK",
            "MESSAGES",
            "fdf*",
            "FORCE_STRESS",
            "TIMES",
            "base_and_mol.*",
            "0_NORMAL_EXIT",
        ]

        for fname in fnames:
            command = f"mv {fname} {savepath}"
            subprocess.run(command, shell=True)

        return float(line.split()[-1])

    def ads(self):
        work_dir = self.work_dir

        base = poscar.poscar(f"target_structure/{self.fname_base}")
        energy_base = self.SIESTA(
            base,
            "base.fdf",
            f"./{work_dir}/{self.method}/ads_base",
            cutoff=self.cutoff,
            PAO=self.PAO,
            is_plus_U=self.is_plus_U,
        )

        mol = poscar.poscar(f"target_structure/{self.fname_mol}")
        mol.adjust_g_to_0()
        mol.h = base.h
        mol.h_inv = base.h_inv
        for ia in range(mol.natm_all):
            mol.ra[ia] = np.dot(mol.h_inv, mol.ta[ia])

        for ia in range(mol.natm_all):
            for id in range(3):
                mol.ra[ia][id] += 0.5
        energy_mol = self.SIESTA(
            mol,
            "mol.fdf",
            f"./{work_dir}/{self.method}/ads_mol",
            ncore=4,
            cutoff=self.cutoff,
            PAO=self.PAO,
            is_plus_U=self.is_plus_U,
        )

        return energy_base, energy_mol

    def main(self, nsearch, coodinate):
        work_dir = self.work_dir
        base = poscar.poscar(f"target_structure/{self.fname_base}")
        mol = poscar.poscar(f"target_structure/{self.fname_mol}")
        base.add_mol(mol, coodinate)

        energy = self.SIESTA(
            base,
            "base_and_mol.fdf",
            f"./{work_dir}/{self.method}/{nsearch}",
            cutoff=self.cutoff,
            PAO=self.PAO,
            is_plus_U=self.is_plus_U,
        )

        return energy
