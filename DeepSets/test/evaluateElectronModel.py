import numpy as np
import glob, os, sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DisappTrksML.DeepSets.architecture import *
from DisappTrksML.DeepSets.ElectronModel import *


def indices_matching_criteria_np(arr, criteria):
    return np.where(criteria(arr))[0]


fileDir = "/data/users/rsantos/train/test/test1/"
model_file = "model.h5"
model_params = {
    "phi_layers": [64, 64, 256],
    "f_layers": [64, 64, 64],
    "track_info_indices": [4, 8, 9, 12],
}
arch = ElectronModel(**model_params)
arch.load_model(fileDir + model_file)

cm = np.zeros((2, 2))
input_dir = "/store/user/rsantos/2022/combined_DYJet/test/"
inputFiles = glob.glob(input_dir + "images_*.root.npz")
inputIndices = np.array([f.split("images_")[-1][:-9] for f in inputFiles])
nFiles = len(inputIndices)
test_files = [
    f"{input_dir}images_{x}.root.npz" for x in inputIndices[int(0.9 * nFiles) :]
]
cm = [[0, 0],
      [0, 0]]
# Loop over all of the test_files
for i, fname in enumerate(test_files):
    _, signal_preds = arch.evaluate_npy(fname, obj=["signal", "signal_info"])
    _, background_preds = arch.evaluate_npy(fname, obj=["background", "background_info"])
    cm[1][1] += np.count_nonzero(signal_preds[:,1] > 0.5)
    cm[0][1] += np.count_nonzero(background_preds[:,1] > 0.5)
    cm[1][0] += np.count_nonzero(signal_preds[:,1] <= 0.5)
    cm[0][0] += np.count_nonzero(background_preds[:,1] <= 0.5)

precision, recall, f1 = arch.calc_binary_metrics(cm)
print("Precision " , precision)
print("Recall ", recall)
print("f1 " , f1)
