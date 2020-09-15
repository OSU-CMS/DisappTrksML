import os
import shutil
import numpy as np

dataDir = "/store/user/llavezzo/disappearingTracks/converted_deepSets100_Zee/"
testDir = "/store/user/llavezzo/disappearingTracks/converted_deepSets100_Zee/test/"

#os.makedirs(testDir)

nFiles = 0
for file in os.listdir(dataDir):
	if(("images" in file) and (".npz" in file)): nFiles+=1

testSet = np.random.randint(0,nFiles-1,int(nFiles*.2))

for file in os.listdir(dataDir):
	if(("images" in file) and (".npz" in file)):

		index = file.find(".")
		fileNum = int(file[11:index])

		if(fileNum in testSet):
			shutil.move(dataDir+file,testDir+file)
