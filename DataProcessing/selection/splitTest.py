import os
import shutil
import numpy as np

dataDir = "/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v4_selection/"
testDir = "/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v4_selection/test/"

os.makedirs(testDir)

files = []
for file in os.listdir(dataDir):
	if(("hist" in file) and (".root" in file)):
		index1 = file.find("_")
		index2 = file.find(".")
		fileNum = int(file[index1+1:index2])
		files.append(fileNum)
nFiles = len(files)
print("Found",nFiles,"files")

testIndices = np.random.randint(0,nFiles-1,int(nFiles*.2))
files = np.array(files)
testSet = files[testIndices]
print("Selecting",len(testSet),"test files")

for file in os.listdir(dataDir):
	if(("hist" in file) and (".root" in file)):

		index1 = file.find("_")
		index2 = file.find(".")
		fileNum = int(file[index1+1:index2])
		
		if(fileNum in testSet):
			shutil.move(dataDir+file,testDir+file)