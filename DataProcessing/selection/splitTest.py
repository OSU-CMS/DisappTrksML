import os
import shutil
import numpy as np

dataDir = "/store/user/llavezzo/disappearingTracks/converted_deepSets100_Zee_V3/"
testDir = "/store/user/llavezzo/disappearingTracks/converted_deepSets100_Zee_V3/test/"

os.makedirs(testDir)

nFiles = 0
for file in os.listdir(dataDir):
	if(("hist" in file) and (".root" in file)): nFiles+=1

testSet = np.random.randint(0,nFiles-1,int(nFiles*.2))

for file in os.listdir(dataDir):
	if(("hist" in file) and (".root" in file)):

		index1 = file.find("_")
		index2 = file.find(".")
		fileNum = int(file[index1+1:index2])

		if(fileNum in testSet):
			shutil.move(dataDir+file,testDir+file)
