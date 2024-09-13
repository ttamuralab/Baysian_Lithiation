import json
import subprocess
import os
import shutil

import stretch

with open("settings.json", "r") as f:
    settings = json.load(f)


max_target_nLi = 189
max_target_nLi = 200

settings["max_target_nLi"] = max_target_nLi
settings["fname_base"] = "POSCAR_Si50O50"
settings["fname_mol"] = "POSCAR_Li"

for i in range(round(max_target_nLi / 10)):
    for j in range(10):
        if i * 10 + j > max_target_nLi:
            target_nLi = max_target_nLi
        else:
            target_nLi = i * 10 + j + 1

        work_dir = f"Li200/Li{(i+1) * 10}/{j}"

        settings["work_dir"] = work_dir
        settings["target_nLi"] = target_nLi

        with open("settings.json", "w") as f:
            json.dump(settings, f, indent=2)

        # os.makedirs(f"{work_dir}")
        shutil.copytree("Li200/base", work_dir, dirs_exist_ok=True)
        # exit()

        if j == 0:
            # *候補点作成
            if not os.path.isfile(f"{work_dir}/candidate/candidate_cooang_t_all.dat"):
                cmd = "python make_candidate.py"
                subprocess.run(cmd, shell=True)

            # *SOAP計算
            if not os.path.isfile(f"{work_dir}/candidate/candidate_descriptor_all.dat"):
                cmd = "python make_soap.py"
                subprocess.run(cmd, shell=True)

            # *PCA
            if not os.path.isfile(f"{work_dir}/candidate/candidate_feature_all.dat"):
                cmd = "python make_feature.py"
                subprocess.run(cmd, shell=True)
                # exit()

        # *BO
        if not os.path.isfile(f"target_structure/POSCAR_Li{target_nLi}"):
            cmd = "python main_BO.py"
            subprocess.run(cmd, shell=True)
            settings["fname_base"] = f"POSCAR_Li{target_nLi}"
            # exit()

        # struct_out_to_poscar.main(
        #     f"{work_dir}/SIESTA/CG/base_and_mol.STRUCT_OUT",
        #     f"target_structure/POSCAR_Li{target_nLi}",
        # )

        # *target_structureを変更
        settings["fname_base"] = f"POSCAR_Li{target_nLi}"
        # exit()
    # exit()
    # *CG
    # CG.main(f"Li189/Li{(i+1) * 10}", (i + 1) * 10)

    # # *最終構造を次の始構造に追加
    # struct_out_to_poscar.main(
    #     f"Li189/Li{(i+1) * 10}/CG/base_and_mol.STRUCT_OUT",
    #     f"target_structure/POSCAR_Li{target_nLi}_CG",
    # )

    # *伸長
    stretch.main(f"POSCAR_Li{target_nLi}")
    settings["fname_base"] = f"POSCAR_Li{target_nLi}_stretched"

import gmail

from_mail = "shogo120922@gmail.com"
password = "ziderdeqfglsgqfu"
to_mail = "shogo120922@gmail.com"
mailText = "program completed"
subject = "notification"
gmail.send(from_mail, password, to_mail, mailText, subject)
