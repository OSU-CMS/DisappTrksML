import os
import numpy as np
import sys
import json
import math
import pickle

dataDir = "/data/disappearingTracks/converted_DYJetsToLL_M50/"
tag = "0p25_"
signal = "e"					#choose: e, m, bkg

"""
infos:	

0: ID
1: matched track gen truth flavor (1: electrons, 2: muons, 0: everything else)
2: nPV
3: deltaRToClosestElectron
4: deltaRToClosestMuon
5: deltaRToClosestTauHaud

"""
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


sCounts = {}
bkgCounts = {}

for fileNum in range(0,4000):

	if(signal == "e"):
		signal_index = [0]
		bkg_index = [1,2]
		reco_index = 3
	if(signal == "m"):
		signal_index = [1]
		bkg_index = [0,2]
		reco_index = 4

	e_fname = 'images_e_'+tag+str(fileNum)+'.npz'
	m_fname = 'images_m_'+tag+str(fileNum)+'.npz'
	bkg_fname = 'images_bkg_'+tag+str(fileNum)+'.npz'

	# check if file exists
	if(not os.path.isfile(dataDir+bkg_fname)): continue

	# import e, m, bkg files containing images and infos
	temp1 = np.load(dataDir+e_fname)
	temp2 = np.load(dataDir+m_fname)
	temp3 = np.load(dataDir+bkg_fname)

	infos = np.asarray([temp1['infos'],temp2['infos'],temp3['infos']])
	images = np.asarray([temp1['images'],temp2['images'],temp3['images']])

	# join the files based on classification 
	s_images = images[signal_index]
	s_infos = infos[signal_index]
	bkg_images = images[bkg_index]
	bkg_infos = infos[bkg_index]

	s_images = np.vstack(s_images)
	s_infos = np.vstack(s_infos)
	bkg_images = np.vstack(bkg_images)
	bkg_infos = np.vstack(bkg_infos)	

	# select signal reco fail, convert to tanh
	s_outImages, s_outInfos = [],[]
	for info, image in zip(s_infos, s_images):
		if(math.fabs(info[reco_index]) > 0.15):
			s_outImages.append(image)
			s_outInfos.append(info)

	# select signal reco fail, convert to tanh
	bkg_outImages, bkg_outInfos = [],[]
	for info, image in zip(bkg_infos, bkg_images):
		if(math.fabs(info[reco_index]) > 0.15):
			bkg_outImages.append(image)
			bkg_outInfos.append(info)

	# some checks before saving
	assert len(s_outImages)==len(s_outInfos)
	assert len(bkg_outImages)==len(bkg_outInfos)

	sCounts.update({fileNum:len(s_outImages)})
	bkgCounts.update({fileNum:len(bkg_outImages)})

	print("File",fileNum)
	print("Saving",len(s_outImages),len(bkg_outImages),"from",len(s_images)+len(bkg_images),"files")
	print()

	# save files	
	outDir = '/data/disappearingTracks/out/'
	f1 = signal+'_'+tag+str(fileNum)
	f2 = 'bkg_'+tag+str(fileNum)

	np.save(outDir+f1,s_outImages)
	np.save(outDir+f2,bkg_outImages)

save_obj(sCounts, "sCounts.pkl")
save_obj(bkgCounts, "bkgCounts.pkl")
