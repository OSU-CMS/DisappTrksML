import numpy as np
import glob, os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from DisappTrksML.DeepSets.architecture import *

# parameters
dataDir = "/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_converted/"
modelFile = 'kfold17_noBatchNorm_finalTrainV3/model.h5'
outDir = "kfold17_noBatchNorm_finalTrainV3/"

# initialize architecture and load the weights/model
arch = DeepSetsArchitecture()
arch.load_model(modelFile)

# iterate over desired files and predict
true = None
preds = None
inputFiles = np.loadtxt(outDir + "/validation_files.txt")

for iFile, file in enumerate(inputFiles):
	print str(iFile)+'/'+str(len(inputFiles))
	if iFile == 10: break

	fname = dataDir+'images_'+str(int(file))+'.root.npz'

	# evaluate file
	for obj in ['signal','background']:

		skip, thisFilePreds = arch.evaluate_npy(fname, track_info=True, obj=obj)
		if skip: continue

		if obj == 'signal': trueThisFile = np.ones(len(thisFilePreds))
		elif obj == 'background': trueThisFile = np.zeros(len(thisFilePreds))

		if true is None:
			true = trueThisFile
			preds = thisFilePreds[:,1]
		else:
			true = np.concatenate((true, trueThisFile))
			preds = np.concatenate((preds, thisFilePreds[:,1]))

# calculate and save metrics
metrics = arch.metrics_per_cut(true, preds, 40)
np.save(outDir+'metrics.npy',metrics)

plt.plot(metrics['splits'])
plt.plot(metrics['precision'], label="precision")
plt.plot(metrics['recall'], label="recall")
plt.plot(metrics['f1'], label="f1")
plt.legend()
plt.savefig(outDir+"metrics.png")