import numpy as np
import glob, os, sys

old = np.load("truthMuons_recoFailed.npz",allow_pickle=True)
oldInfos = old['infos'][:,:3]

infos = None
bkgDir = "/store/user/llavezzo/disappearingTracks/nonRecoGenMuons_v6/"
inputFiles = glob.glob(bkgDir+'images_*.root.npz')
inputIndices = np.array([f.split(bkgDir+'images_')[-1][:-9] for f in inputFiles])
for i in range(len(inputFiles)):
	data = np.load(bkgDir+'images_'+str(inputIndices[i])+'.root.npz',allow_pickle=True)
	if(infos is None): infos = data['signal_info'][:,:3]
	else: infos = np.vstack((infos, data['signal_info'][:,:3]))
np.savez_compressed("nonRecoGenMuons_v6_infos.npz",infos = infos)
nfound = 0
for oinfo in oldInfos:
	found = False
	for ninfo in infos:
		if ninfo[0] == oinfo[0] and ninfo[1] == oinfo[1] and ninfo[2] == oinfo[2]: found = True 

	if not found: 
		print(oinfo)
	else:
		nfound +=1 
print("found",nfound)
print(len(oldInfos))
print(len(infos))