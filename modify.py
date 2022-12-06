import copy
import cv2
import h5py
from tqdm import tqdm
import numpy as np
import os

base_path = "Dataset/BJCells_ACTD_08_26_2022"
PWS_Path = f"{base_path}/PWSimages"
DV_Path = f"{base_path}/DVimages"

for folder in tqdm(os.listdir(DV_Path)):
    # print("Current folder:", folder)
    if len(folder) < 6:
        old_name = os.path.join(DV_Path, folder)
        new_name = os.path.join(DV_Path, f"{folder[0:4]}0{folder[-1]}")
        os.rename(old_name, new_name)

for folder in tqdm(os.listdir(PWS_Path)):
    # print("Current folder:", folder)
    if len(folder) < 6:
        old_name = os.path.join(PWS_Path, folder)
        new_name = os.path.join(PWS_Path, f"{folder[0:4]}0{folder[-1]}")
        os.rename(old_name, new_name)

for folder in os.listdir(DV_Path):
    print("Current folder:", folder)
    DV = os.path.join(DV_Path, folder)
    for file in os.listdir(DV):
        if ".tif" in file:
            old_name = os.path.join(DV_Path, folder, file)
            new_name = os.path.join(DV_Path, folder, "R3D_D3D-processed.tif")
            os.rename(old_name, new_name)
        elif "label" in file:
            if not os.path.exists(os.path.join(DV_Path, folder, "labels.mat")):
                old_name = os.path.join(DV_Path, folder, file)
                new_name = os.path.join(DV_Path, folder, "labels.mat")
                os.rename(old_name, new_name)
