from utils_pre import *
from tqdm import tqdm

base_path = "Dataset/BJCells_ACTD_08_26_2022"
PWS_Path = f"{base_path}/PWSimages"
DV_Path = f"{base_path}/DVimages"
Result_Path = f"{base_path}/Results"
dataset_pws_min = np.array([])
dataset_pws_max = np.array([])
dataset_dv_min = np.array([])
dataset_dv_max = np.array([])

for folder in tqdm(os.listdir(DV_Path)):
    pre = preprocess(PWS_Path, DV_Path, instance=folder)
    if not pre.data_availability(folder):
        continue

    pre.preprocessing_pipline(n=3,
                              output_size=[512, 512],
                              level_iters=[25],
                              inv_iter=50,
                              )
    dataset_pws_min = np.append(dataset_pws_min, pre.cropped_pws.min())
    dataset_pws_max = np.append(dataset_pws_max, pre.cropped_pws.max())
    dataset_dv_min = np.append(dataset_dv_min, pre.cropped_dv.min())
    dataset_dv_max = np.append(dataset_dv_max, pre.cropped_dv.max())
dataset_range = [np.max(dataset_pws_max) - np.min(dataset_pws_min),
                 np.max(dataset_dv_max) - np.min(dataset_dv_min)]
os.remove(pre.diagnose_txt)


for folder in tqdm(os.listdir(DV_Path)):
    cell = preprocess(PWS_Path, DV_Path, instance=folder)
    if not cell.data_availability(folder):
        continue
    cell.preprocessing_pipline(n=3,
                               output_size=[512, 512],
                               level_iters=[200, 100, 50, 25],
                               inv_iter=50,
                               norm_range=dataset_range,
                               save=True
                               )
