#!/usr/bin/env python

import os
import sys
import math
import glob
import pickle
from ROOT import TFile, TTree, gROOT
from ROOT.Math import XYZVector
import numpy as np
import time

######## parameters ################################################################
dataDir = ''
eta_range = 0.3
phi_range = 0.3
maxHitsInImages = 100
####################################################################################

# script arguments
fileNum = int(sys.argv[1])
if(len(sys.argv)>2): 
	dataDir = str(sys.argv[2])
	if(len(sys.argv)==4):
		fileList = str(sys.argv[3])
		inarray = np.loadtxt(fileList,dtype=float)
		fileNum = int(inarray[fileNum])
fname = "hist_"+str(fileNum)+".root"
print "File "+dataDir+fname 

# combine EB+EE and muon detectors into ECAL/HCAL/MUO indices
def detectorIndex(detType):
	if detType == 1 or detType == 2:
		return 0
	elif detType == 4:
		return 1
	elif detType >= 5 and detType <= 7:
		return 2
	else:
		return -1

# return (dEta, dPhi) between track and hit
def imageCoordinates(track, hit):
	momentum = XYZVector(track.px, track.py, track.pz)
	track_eta = momentum.Eta()
	track_phi = momentum.Phi()
	dEta = track_eta - hit.eta
	dPhi = track_phi - hit.phi
	# branch cut [-pi, pi)
	if abs(dPhi) > math.pi:
		dPhi -= round(dPhi / (2. * math.pi)) * 2. * math.pi
	return (dEta, dPhi)

fin = TFile(dataDir+fname, 'read')
eTree = fin.Get('eTree')
bTree = fin.Get('bTree')

signal, s_infos = [], []
bkg, bkg_infos = [], []

for class_label,tree in zip([0,1],[bTree,eTree]):

	for event in tree:
		nPV = event.nPV
		eventNumber = event.eventNumber
		lumiBlockNumber = event.lumiBlockNumber
		runNumber = event.runNumber
		trackNumber = 0

		for track in event.tracks:
			
			hits = []
			for hit in event.recHits:
				dEta, dPhi = imageCoordinates(track, hit)
				if abs(dEta) < eta_range and abs(dPhi) < phi_range:
					detIndex = detectorIndex(hit.detType)
					if detIndex < 0: continue
					energy = hit.energy if detIndex != 2 else 1
					hits.append((dEta, dPhi, energy, detIndex))

			if(len(hits) > 0):
				hits = np.reshape(hits, (len(hits),4))
				hits = hits[hits[:,2].argsort()]
				hits = np.flip(hits, axis=0)
				assert np.max(hits[:,2])==hits[0,2]

			momentum = XYZVector(track.px,track.py,track.pz)
			track_eta = momentum.Eta()
			track_phi = momentum.Phi()

			sets = np.zeros((maxHitsInImages,4))
			for iHit in range(min(len(hits), maxHitsInImages)):
				sets[iHit][0] = hits[iHit][0]
				sets[iHit][1] = hits[iHit][1]
				sets[iHit][2] = hits[iHit][2]
				sets[iHit][3] = hits[iHit][3]

			sets = np.concatenate(([eventNumber, lumiBlockNumber, runNumber, trackNumber], sets.flatten().astype('float32')))
			infos = np.array([
					eventNumber, lumiBlockNumber, runNumber, trackNumber,
					class_label,
					nPV,
					track.deltaRToClosestElectron,
					track.deltaRToClosestMuon,
					track.deltaRToClosestTauHad,
					track_eta,
					track_phi,
					track.dRMinBadEcalChannel,
					track.nLayersWithMeasurement
				])

			if(class_label == 0):
				signal.append(sets)
				s_infos.append(infos)
			if(class_label == 1):
				bkg.append(sets)
				bkg_infos.append(infos)

			trackNumber+=1
		
np.savez_compressed('events_'+str(fileNum)+'.npz', 
					signal=signal,
					bkg=bkg,
					s_infos=s_infos,
					bkg_infos=bkg_infos)