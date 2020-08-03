import os
import numpy as np
import sys
import json
import math

dataDir = "/store/user/llavezzo/disappearingTracks/converted_DYJetsToLL_M50/"
outDataDir = "/store/user/llavezzo/disappearingTracks/electron_selection/"
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

if(signal == "e"):
	signal_index = [0]
	bkg_index = [1,2]
	reco_index = 3
if(signal == "m"):
	signal_index = [1]
	bkg_index = [0,2]
	reco_index = 4
	
# keep count of how many signal, bkg events per file
sCounts = {}
bkgCounts = {}

for i in range(4000):

	if(i < 185): continue

	e_fname = 'images_e_'+tag+str(i)+'.npz'
	m_fname = 'images_m_'+tag+str(i)+'.npz'
	bkg_fname = 'images_bkg_'+tag+str(i)+'.npz'

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
			s_outImages.append(np.append(image[0],np.tanh(image[1:])))
			s_outInfos.append(info)

	# select signal reco fail, convert to tanh
	bkg_outImages, bkg_outInfos = [],[]
	for info, image in zip(bkg_infos, bkg_images):
		if(math.fabs(info[reco_index]) > 0.15):
			bkg_outImages.append(np.append(image[0],np.tanh(image[1:])))
			bkg_outInfos.append(info)

	# some checks before saving
	assert len(s_outImages)==len(s_outInfos)
	assert len(bkg_outImages)==len(bkg_outInfos)

	# update counting dictionaries
	sCounts.update({i : len(s_outImages)})
	bkgCounts.update({i : len(bkg_outImages)})

	print("File",i)
	print("Saving",len(s_outImages),len(bkg_outImages),"from",len(s_images)+len(bkg_images),"files")
	print()


	# save and move to appropriate dir
	f1 = signal+'_'+tag+"tanh_"+str(i)
	f2 = 'bkg_'+tag+"tanh_"+str(i)
	np.savez_compressed(f1,images=s_outImages,infos=s_outInfos)
	np.savez_compressed(f2,images=bkg_outImages,infos=bkg_outInfos)
	np.save(f1,s_outImages)
	np.save(f2,bkg_outImages)
	os.system("mv "+f1+".npz "+outDataDir+f1+".npz")
	os.system("mv "+f2+".npz "+outDataDir+f2+".npz")
	os.system("mv "+f1+".npy "+outDataDir+f1+".npy")
	os.system("mv "+f2+".npy "+outDataDir+f2+".npy")

# save the dictionaries
with open(signal+'SignalCounts.json', 'w', encoding='utf-8') as f:
    json.dump(sCounts, f, ensure_ascii=False, indent=4)
with open(signal+'BackgroundCounts.json', 'w', encoding='utf-8') as f:
    json.dump(bkgCounts, f, ensure_ascii=False, indent=4)