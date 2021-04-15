import numpy as np
import glob, os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from DisappTrksML.DeepSets.DisappearingTracksAnalysis import *

def fiducialMapSigma(eta, phi, fiducial_maps):
    maxSigma = -999.

    iBin = fiducial_maps['ele'].GetXaxis().FindBin(eta)
    jBin = fiducial_maps['ele'].GetYaxis().FindBin(phi)

    for i in range(iBin-1, iBin+2):
        if i > fiducial_maps['ele'].GetNbinsX() or i < 0: continue
        for j in range(jBin-1, jBin+2):
            if j > fiducial_maps['ele'].GetNbinsY() or j < 0: continue

            dEta = eta - fiducial_maps['ele'].GetXaxis().GetBinCenter(i)
            dPhi = phi - fiducial_maps['ele'].GetYaxis().GetBinCenter(j)

            if dEta*dEta + dPhi*dPhi > 0.05*0.5:
                continue
            
            if fiducial_maps['ele'].GetBinContent(i, j) > maxSigma:
                maxSigma = fiducial_maps['ele'].GetBinContent(i, j)

            if fiducial_maps['muon'].GetBinContent(i, j) > maxSigma:
                maxSigma = fiducial_maps['muon'].GetBinContent(i, j)

    return maxSigma

payload_dir = '/share/scratch0/llavezzo/CMSSW_11_1_3/src/OSUT3Analysis/Configuration/data/'
fiducial_maps_2017F = calculateFidicualMaps(
    payload_dir + 'electronFiducialMap_2017_data.root',
    payload_dir + 'muonFiducialMap_2017_data.root',
    '_2017F')
# fiducial_maps_mc = calculateFidicualMaps(
#     payload_dir + 'electronFiducialMap_2017_data.root',
#     payload_dir + 'muonFiducialMap_2017_data.root',
#    '')

dirs = ["/store/user/llavezzo/disappearingTracks/electronsTesting/SingleEle_fullSel_pt1_FIXED/",
		"/store/user/llavezzo/disappearingTracks/electronsTesting/SingleEle_fullSel_pt2_FIXED/"]
inputFiles = []
for d in dirs: inputFiles += glob.glob(d+'*.root.npz')

sigmas = []
for i,fname in enumerate(inputFiles):
	print(i)
	infile = np.load(fname, allow_pickle=True)['infos']

	for track in infile:
		sigmas.append(fiducialMapSigma(track[8], track[9], fiducial_maps_2017F))

np.save("ele_fidMap.npy", sigmas)