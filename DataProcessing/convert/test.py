import numpy as np
import os

dataDir = "/store/user/llavezzo/disappearingTracks/AMSB_800GeV_10000cm_sets_fullSel_noEcaloCut/"
count = 0
for file in os.listdir(dataDir):
	if not("events" in file and ".npz" in file): continue
	thisFile = np.load(dataDir+file)['signal']
	count += thisFile.shape[0]
print(count)