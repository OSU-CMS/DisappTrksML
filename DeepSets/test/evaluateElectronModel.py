import numpy as np
import glob, os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from architecture import *
from ElectronModel import *

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
