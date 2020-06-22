import numpy as np
import os
from utils import load_all_data

dataDir = '/store/user/llavezzo/images/'
tag = '_0p25_tanh'
fOut = 'electron_selection_0p25_tanh.npz'
pos_class = [1]
neg_class = [0,2]

images, infos = load_all_data(dataDir, tag)

outImages =[]
outInfos = []
nElectrons = 0
for i in range(len(images)):
	if(infos[i,3] > 0.15):

		if(len(outImages)%1000==0): print(len(outImages))

		# Chek IDs match, and drop ID from image
		assert images[i,-1] == infos[i,0], "Image and Info IDS don't match"
		outImages.append(images[i,:])
		
		info = infos[i]
		if(int(info[1]) in pos_class): 
			info = np.append(info,1)
			nElectrons+=1
		if(info[1] in neg_class): info = np.append(info,0)
		outInfos.append(info)

assert len(outImages) == len(outInfos), "Out images and out infos don't match"
print("nElectrons",nElectrons)
outImages = np.asarray(outImages)
np.savez_compressed(fOut,images=outImages,infos=outInfos)
os.system('mv '+str(fOut)+' '+dataDir+fOut)