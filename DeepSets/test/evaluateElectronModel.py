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


