"""
Counts the number of events from a given folder.
Use it to get an idea how many events pass a selection in each class 
after using the condorized run_convert.py.

Arguments:
Pass the name of each class in the .npz file in the 'objects' array
and the directory as the dataDir.
"""

import os, sys
import numpy as np
import pickle as pkl

dataDir = "/store/user/llavezzo/disappearingTracks/muonsTesting/SingleMuon_pt1/"
objects = ["tracks"]
class_count = np.zeros(len(objects))

for file in os.listdir(dataDir):

	if("images" not in file): continue 
	if(".root.npz" not in file): continue 

	print(file)

	fin = np.load(dataDir+file, allow_pickle=True)

	for iClass, obj in enumerate(objects):
		class_count[iClass] += int(fin[obj].shape[0])
	
for obj, count in zip(objects, class_count):
	print((obj, "\t", count))