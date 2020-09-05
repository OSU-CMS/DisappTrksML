import os
import numpy as np
import sys
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


dataDir = "/store/user/mcarrigan/disappearingTracks/converted_DYJetsToLL_M50_V3/"

requireThreshold = False    # require <count> images above <energy> when requireThreshold is true
count = 5
energy = 0.5
featureSelection = True    # engineered features selection 
failAllRecos = False         # require events to fail all reconstructions

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


# both fail selections
electronsTotal = []
electronsFailedEReco = []

for file in os.listdir(dataDir):

	if("images_e_" not in  file): continue

	# import images and infos
	temp= np.load(dataDir+file)
	infos = temp['infos']
	images = temp['images']    

	for info, image in zip(infos, images):

		# fail feature selection
		if(featureSelection and not passesSelection(image)): continue	

		electronsTotal.append([info[7],info[8]])

		# fail electron reco
		if(math.fabs(info[4]) < 0.15): continue

		electronsFailedEReco.append([info[7],info[8]])

	print(file)


np.save("electronsTotal",electronsTotal)
np.save("electronsFailedEReco",electronsFailedEReco)