import glob, os, sys
import math
import numpy as np

fileDir = "/store/user/llavezzo/disappearingTracks/SingleMu_2017F_wIso/"
inputFiles = glob.glob(fileDir+'images_*.root.npz')
bkg, signal = 0, 0
for i,fname in enumerate(inputFiles):
	data = np.load(fname, allow_pickle=True)

	bkg += data['sets'].shape[0]
	# signal += data['signal'].shape[0]

	print("bkg",bkg)
	# print("signal",signal)
	