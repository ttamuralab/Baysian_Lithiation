import subprocess
import poscar_to_fdf
import os
import poscar


def main(ifname, savepath, nb=1):
    # run CG
    # copy base_dir to current
    command = f"cp -r ./Li200/base/SIESTA/base/* ."
    subprocess.run(command, shell=True)

    fname_fdf = f"CG.fdf"
    base = poscar.poscar(ifname)
    base.h[1][1] = base.h[1][1] * nb

    poscar_to_fdf.main(base, f"./{fname_fdf}", cgstep=1000)

    # run CG
    ncore = 24
    cmd = f"mpirun -np {ncore} siesta -fdf XML.Write {fname_fdf} > log_SIESTA"
    subprocess.run(cmd, shell=True)
    os.makedirs(
        savepath,
        exist_ok=True,
    )
    # move strage dir
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

    print("finish CG")
