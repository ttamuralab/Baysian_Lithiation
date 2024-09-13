import numpy as np
import json

from Lib import pca

with open("settings.json", "r") as f:
    settings = json.load(f)
work_dir = settings["work_dir"]

print("program make_feature.py")
print("work_dir =", work_dir)

pca.main("candidate_descriptor_all.dat", "candidate_feature_all.dat", work_dir, 15)
