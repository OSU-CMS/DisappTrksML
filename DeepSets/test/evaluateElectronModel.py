import numpy as np
import glob, os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
#from DisappTrksML.DeepSets.architecture import *
from DisappTrksML.DeepSets.ElectronModel import *
DATAPOINTS = 2
REGEN = True

parent_directory = "train_backup/"
data_directory = "/store/user/rsantos/2022/combined_DYJet/test/"
model = ElectronModel()
model.load_model(parent_directory + "model.h5", compile=False)
print("Loaded model!") 

# with os.scandir(parent_directory) as it:
#     model = ElectronModel()
#     for subdirectory in it: 
#         if os.path.isdir(subdirectory): 
#             print("Path: " , os.path.join(subdirectory, "model.h5"))
#             model.load_model(os.path.join(subdirectory, "model.h5"))

#             cm = np.zeros((2,2))
#             model_values = np.zeros((DATAPOINTS,4))

#             if REGEN:
#                 for i, thresh in enumerate(np.linspace(0, 1, num=DATAPOINTS, endpoint=False)):
#                     metrics = model.get_metrics(input_dir=data_directory, threshold= thresh, glob_pattern="images_*.root.npz")
#                     accuracy = (metrics[0] + metrics[1]) / (metrics[0] + metrics[1] + metrics[2] + metrics[3])
#                     recall = metrics[0]/(metrics[0] + metrics[2])
#                     precision = metrics[0]/(metrics[0] + metrics[3])
#                     f1 = 2 * ((precision * recall)/(precision + recall))
#                     model_values[i] =  [accuracy, recall, precision, f1]
#                     np.save("metric", model_values)

#             else:
#                 metrics = np.load("metric.npy")
#                 plt.plot(np.linspace(0, 1, num=DATAPOINTS, endpoint=False), metrics[:, 3])
#                 plt.xlabel("Classification Threshold")
#                 plt.ylabel("F1 Score")
#                 plt.title("F1 Score for Electron Classifier")

#                 plt.savefig("f1.png")


# initialize the model with the weights
#fileDir = '/data/users/llavezzo/forBrian/kfold19_noBatchNorm_finalTrainV3/'
fileDir = '/home/rsantos/scratch0/CMSSW_12_4_11_patch3/src/DisappTrksML/DeepSets/test/train_2023-08-20_21.31.24/'
model_file = 'model.h5'
model_params = {
	'phi_layers':[400,256,128], 
	'f_layers': [128,128,64,32],
	'track_info_indices' : [4,8,9,12]
}
arch = ElectronModel(**model_params)
arch.load_model(fileDir+model_file)

cm = np.zeros((2,2))

# evaluate the model
arch.evaluate_dir(["/store/user/rsantos/2022/combined_DYJet/test/"])
np.save("SingleEle_fullSel_preds.npy", totPreds)
