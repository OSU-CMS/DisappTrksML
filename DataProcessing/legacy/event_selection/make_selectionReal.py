import os
import numpy as np
import sys
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


dataDir = "/store/user/mcarrigan/disappearingTracks/AMSB/converted_600_1000_step3_tanh/"
tag = "0p25_"
signal = "e"					#choose: e, m, bkg

if len(sys.argv) >= 2: dataDir = str(sys.argv[2])

requireThreshold = False	# require <count> images above <energy> when requireThreshold is true
count = 5
energy = 0.5
featureSelection = False 	# engineered features selection 
failAllRecos = True 		# require events to fail all reconstructions
scale = False 				# scaling

"""
infos:	
0: File
1: ID
2: matched track gen truth flavor (1: electrons, 2: muons, 0: everything else)
3: nPV
4: deltaRToClosestElectron
5: deltaRToClosestMuon
6: deltaRToClosestTauHaud
7: eta
8: phi
9: genID
10: genDR
11: pt
"""

def thresholdEnergies(matrix, pcount, penergy):
    aboveThreshold = False
    nonZero = np.nonzero(matrix > penergy)
    if len(nonZero[0]) >= pcount: aboveThreshold = True
    print("count ", len(nonZero[0]), "passes selection ", aboveThreshold)
    return aboveThreshold 

def passesSelection(img):

	img = np.reshape(img[1:],[40,40,4])
	img = img[:,:,0]

	coords = []
	for i in range(3):

		# at least 3 non zero hits
		if(img.max()==0): 
			return False

		if i == 0:

			# energy of largest pixel
			if(img.max() < 0.5): return False

			# energy around max
			indices = np.where(img == img.max())
			activityAroundMax = 0
			for i in range(indices[0][0]-10,indices[0][0]+10+1):
				for j in range(indices[1][0]-10,indices[1][0]+10+1):
					if(i > 39 or j > 39): continue
					if(i==indices[0][0] and j==indices[1][0]): continue
					activityAroundMax+=img[i,j]
			if(activityAroundMax < 1): return False

		# store indices of largest 3 hits
		indices = np.where(img == img.max())
		if(len(indices[0]) > 1): indices =  np.array([indices[0][0],indices[1][0]])
		coords.append(indices)
		img[indices] = 0

	# radius veto on 3 largest hits
	if(len(coords)==3):
		a = math.sqrt(pow(coords[0][0]-coords[1][0], 2)+pow(coords[0][1]-coords[1][1],2)) 
		b = math.sqrt(pow(coords[0][0]-coords[2][0], 2)+pow(coords[0][1]-coords[2][1],2)) 
		c = math.sqrt(pow(coords[1][0]-coords[2][0], 2)+pow(coords[1][1]-coords[1][1],2)) 
		d = np.mean([a,b,c])
		if(d > 20): return False
	else:
		return False

	return True

if(signal == "e"):
	signal_index = [0]
	bkg_index = [1]
	reco_index = [4]
	if(failAllRecos): reco_index = [4,5,6]
#if(signal == "m"):
#	signal_index = [1]
#	bkg_index = [0,2]
#	reco_index = [5]
#	if(failAllRecos): reco_index = [4,5,6]

# script arguments
process = int(sys.argv[1])
print("Process",process)
# name of the file to import
files = np.load('fileslist.npy')
fileNum = files[process]

e_fname = 'images_e_'+tag+str(fileNum)+'.npz'
#m_fname = 'images_m_'+tag+str(fileNum)+'.npz'
bkg_fname = 'images_bkg_'+tag+str(fileNum)+'.npz'

# check if file exists
if(not os.path.isfile(dataDir+bkg_fname)): sys.exit(0)

# import e, m, bkg files containing images and infos
temp1 = np.load(dataDir+e_fname)
#temp2 = np.load(dataDir+m_fname)
temp2 = np.load(dataDir+bkg_fname)

infos = np.asarray([temp1['infos'],temp2['infos']])
images = np.asarray([temp1['images'],temp2['images']])

# join the files based on classification 
s_images = images[signal_index]
s_infos = infos[signal_index]
bkg_images = images[bkg_index]
bkg_infos = infos[bkg_index]

s_images = np.vstack(s_images)
s_infos = np.vstack(s_infos)
bkg_images = np.vstack(bkg_images)
bkg_infos = np.vstack(bkg_infos)	

# apply selections to signal
s_outImages, s_outInfos = [],[]
if len(s_infos) > 0:
	for info, image in zip(s_infos, s_images):
		passReco = False
		for i in reco_index: 
			if(math.fabs(info[i]) < 0.15): passReco = True
		if(passReco): continue
		if(requireThreshold): 
			if(thresholdEnergies(image[1:].flatten(), count, energy) == False): continue
		if(featureSelection and not passesSelection(image)): continue
		if(scale): image = np.tanh(scale)
		s_outImages.append(np.concatenate(([fileNum],image)))
		s_outInfos.append(info)


# apply selections to background
bkg_outImages, bkg_outInfos = [],[]
for info, image in zip(bkg_infos, bkg_images):
	passReco = False
	for i in reco_index: 
		if(math.fabs(info[i]) < 0.15): passReco = True
	if(passReco): continue
	if(requireThreshold): 
		if(thresholdEnergies(image[1:].flatten(), count, energy) == False): continue
	if(featureSelection and not passesSelection(image)): continue
	if(scale): image = np.tanh(scale)
	bkg_outImages.append(np.concatenate(([fileNum],image)))
	bkg_outInfos.append(info)
	debug.append(info[10])

# some checks before saving
assert len(s_outImages)==len(s_outInfos)
assert len(bkg_outImages)==len(bkg_outInfos)

print("File",fileNum)
print("Saving",len(s_outImages),len(bkg_outImages),"from",len(s_images)+len(bkg_images),"files")
print()

# save files	
f1 = signal+'_'+tag+str(fileNum)
f2 = 'bkg_'+tag+str(fileNum)
np.save(f1,s_outImages)
np.save(f2,bkg_outImages)

np.savez_compressed(f1,images=s_outImages,infos=s_outInfos)
np.savez_compressed(f2,images=bkg_outImages,infos=bkg_outInfos)

# fout = open("ThresholdCounts.txt", w)
# fout.write("Signal Passing: " + len(s_outImages) + " out of " + len(s_images))
# fout.write("Background Passing: " + len(bkg_outImages) + " out of " + len(bkg_images))
# fout.close()
