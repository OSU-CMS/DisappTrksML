import os
import numpy as np
import sys
import pickle as pkl
import ROOT as r

dataDir = "/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v4_selection/"

eCounts = {}
bkgCounts = {}
electrons, bkg = 0, 0

for file in os.listdir(dataDir):

	if("hist" not in file): continue 
	if(".root" not in file): continue 

	index1 = file.find("_")
	index2 = file.find(".")
	fileNum = int(file[index1+1:index2])

	fin = r.TFile(dataDir+file, 'read')
	eTree = fin.Get('eTree')
	bTree = fin.Get('bTree')
	electrons_thisTree = eTree.GetEntriesFast()
	bkg_thisTree = bTree.GetEntriesFast()

	eCounts.update({fileNum:electrons_thisTree})
	electrons += electrons_thisTree

	bkgCounts.update({fileNum:bkg_thisTree})
	bkg += bkg_thisTree
	
print("electrons",electrons)
print("bkg",bkg)
with open('eCounts.pkl', 'wb') as f:
	pkl.dump(eCounts,f)
with open('bkgCounts.pkl', 'wb') as f:
	pkl.dump(bkgCounts,f)
