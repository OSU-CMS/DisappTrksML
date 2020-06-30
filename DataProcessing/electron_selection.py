import os
import numpy as np
import sys
import json

dataDir = "/store/user/llavezzo/disappearingTracks/converted_DYJetsToLL_M50/"
outDataDir = "/store/user/llavezzo/disappearingTracks/electron_selection_DYJetsToll_M50/"
tag = "0p25_"

"""
infos:

0: ID
1: matched track gen truth flavor (1: electrons, 2: muons, 0: everything else)
2: nPV
3: deltaRToClosestElectron
4: deltaRToClosestMuon
5: deltaRToClosestTauHaud

"""

# keep count of how many electrons, bkg events per file
eCounts = {}
bkgCounts = {}

for i in range(4000):

	bkg_images, e_images = [],[]
	bkg_infos, e_infos = [],[]

	e_fname = 'images_e_'+tag+str(i)+'.npz'
	m_fname = 'images_m_'+tag+str(i)+'.npz'
	bkg_fname = 'images_bkg_'+tag+str(i)+'.npz'

	# check if file exists
	if(not os.path.isfile(dataDir+e_fname)): continue

	# import e, m, bkg files containing images and infos
	temp1 = np.load(dataDir+e_fname)
	temp2 = np.load(dataDir+m_fname)
	temp3 = np.load(dataDir+bkg_fname)

	infos1 = temp1['infos']
	infos2 = temp2['infos']
	infos3 = temp3['infos']

	images1 = temp1['images']
	images2 = temp2['images']
	images3 = temp3['images']

	# join the files based on classification 
	images_e = images1
	images_bkg = np.concatenate((images2,images3))
	infos_e = infos1
	infos_bkg = np.concatenate((infos2,infos3))

	# select electron reco fail, convert to tanh
	outImages_e, outInfos_e = [],[]
	for info, image in zip(infos_e, images_e):
		if(info[3] < 0.15): continue
		outImages_e.append(np.tanh(image))
		outInfos_e.append(info)

	# select electron reco fail, convert to tanh
	outImages_bkg, outInfos_bkg = [],[]
	for info, image in zip(infos_bkg, images_bkg):
		if(info[3] < 0.15): continue
		outImages_bkg.append(np.tanh(image))
		outInfos_bkg.append(info)

	# some checks before saving
	assert len(outImages_e)==len(outInfos_e)
	assert len(outImages_bkg)==len(outInfos_bkg)

	# update countnig dictionaries
	eCounts.update({i : len(outImages_e)})
	bkgCounts.update({i : len(outImages_bkg)})

	print("File",i)
	print("Saving",len(outImages_e),len(outImages_bkg),"from",len(images1)+len(images2)+len(images3),"files")
	print()


	# save and move to appropriate dir
	f1 = 'e_'+tag+"tanh_"+str(i)
	f2 = 'bkg_'+tag+"tanh_"+str(i)
	np.savez_compressed(f1,images=outImages_e,infos=outInfos_e)
	np.savez_compressed(f2,images=outImages_bkg,infos=outInfos_bkg)
	os.system("mv "+f1+".npz "+outDataDir+f1+".npz")
	os.system("mv "+f2+".npz "+outDataDir+f2+".npz")

# save the dictionaries
with open('eCounts.json', 'w', encoding='utf-8') as f:
    json.dump(eCounts, f, ensure_ascii=False, indent=4)
with open('bkgCounts.json', 'w', encoding='utf-8') as f:
    json.dump(bkgCounts, f, ensure_ascii=False, indent=4)