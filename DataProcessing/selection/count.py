import os
import numpy as np
import sys
import pickle as pkl
import ROOT as r

dataDir = "/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v4_selection_electrons/"

eCounts = {}
bkgCounts = {}
signal, bkg = 0, 0

for file in os.listdir(dataDir):

	if("hist" not in file): continue 
	if(".root" not in file): continue 

	index1 = file.find("_")
	index2 = file.find(".")
	fileNum = int(file[index1+1:index2])

	fin = r.TFile(dataDir+file, 'read')
	sTree = fin.Get('sTree')
	bTree = fin.Get('bTree')
	signal_thisTree = int(sTree.GetEntries())
	bkg_thisTree = int(bTree.GetEntries())

	eCounts.update({fileNum:signal_thisTree})
	signal += signal_thisTree

	bkgCounts.update({fileNum:bkg_thisTree})
	bkg += bkg_thisTree
	
print("signal",signal)
print("bkg",bkg)
with open(dataDir+'sCounts.pkl', 'wb') as f:
	pkl.dump(eCounts,f)
with open(dataDir+'bkgCounts.pkl', 'wb') as f:
	pkl.dump(bkgCounts,f)
