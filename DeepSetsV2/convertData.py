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

gROOT.ProcessLine('.L Infos.h+')

# script arguments
process = int(sys.argv[1])
print("Process",process)

# name of the file to import
try:
	files = np.load('fileslist.npy')
	fileNum = files[process]
except:
	fileNum = process
fname = "hist_"+str(fileNum)+".root"
print("File",fname)

# output file tag
fOut = '_0p5_'+str(fileNum)

######## parameters ################################################################
dataDir = '/store/user/mcarrigan/disappearingTracks/images_DYJetsToLL_M50/'
eta_range = 0.5
phi_range = 0.5
dataMode = False # if true, use tagProbe selection; if false, use genmatched pdgID
maxHitsInImages = 100
####################################################################################

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

# track is close to a reconstructed lepton of flavor
def isReconstructed(track, flavor):
	if flavor == 'ele':
		return abs(track.deltaRToClosestElectron) < 0.15
	elif flavor == 'muon':
		return abs(track.deltaRToClosestMuon) < 0.15
	elif flavor == 'tau':
		return abs(track.deltaRToClosestTauHad) < 0.15
	else:
		return False

# track is gen-matched to pdgId
def isGenMatched(track, pdgId):
	return (abs(track.genMatchedID) == pdgId and abs(track.genMatchedDR) < 0.1)

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

def passesSelection(track):

	momentum = XYZVector(track.px, track.py, track.pz)
	eta = momentum.Eta()
	pt = math.sqrt(momentum.Perp2())

	if not abs(eta) < 2.4: return False
	if track.inGap: return False
	if not abs(track.dRMinJet) > 0.5: return False
	return True

fin = TFile(dataDir+fname, 'read')
tree = fin.Get('trackImageProducer/tree')

e_imgs, bkg_imgs = [], []
e_infos, bkg_infos = [], []
IDe,IDb=0,0

for iTrack, event in enumerate(tree):

	nPV = event.nPV

	for track in event.tracks:
		
		if(not passesSelection(track)): continue

		imageHits = []
		for hit in event.recHits:
			dEta, dPhi = imageCoordinates(track, hit)
			if abs(dEta) < eta_range and abs(dPhi) < phi_range:
				detIndex = detectorIndex(hit.detType)
				if detIndex < 0:
					continue
				energy = hit.energy if detIndex != 2 else 1
				imageHits.append((dEta, dPhi, energy, detIndex))

		if(len(imageHits) > 0):
			imageHits = np.reshape(imageHits, (len(imageHits),4))
			imageHits = imageHits[imageHits[:,2].argsort()]
			imageHits = np.flip(imageHits, axis=0)
			assert np.max(imageHits[:,2])==imageHits[0,2]

		momentum = XYZVector(track.px,track.py,track.pz)
		track_eta = momentum.Eta()
		track_phi = momentum.Phi()

		# fail all recos
		if ((not isReconstructed(track, 'ele')) and 
			(not isReconstructed(track, 'muon')) and
			(not isReconstructed(track, 'tau'))): 

			# truth electron that failed electron reco
			if((dataMode and track.isTagProbeElectron) or ((not dataMode) and isGenMatched(track,11))):
				img = np.zeros((maxHitsInImages,4))
				for iHit in range(min(len(imageHits), maxHitsInImages)):
					img[iHit][0] = imageHits[iHit][0]
					img[iHit][1] = imageHits[iHit][1]
					img[iHit][2] = imageHits[iHit][2]
					img[iHit][3] = imageHits[iHit][3]
					
				e_imgs.append(np.concatenate(([fileNum, IDe],img.flatten())))
				e_infos.append(np.array([
					fileNum,
					IDe,
					1,
					nPV,
					track.deltaRToClosestElectron,
					track.deltaRToClosestMuon,
					track.deltaRToClosestTauHad,
					track_eta,
					track_phi,
					track.genMatchedID,
					track.genMatchedDR
				]))
				IDe+=1

			# truth non-electrons that failed all recos
			elif((dataMode and (not track.isTagProbeElectron)) or ((not dataMode) and (not isGenMatched(track,11)))):
				img = np.zeros((maxHitsInImages,4))
				for iHit in range(min(len(imageHits), maxHitsInImages)):
					img[iHit][0] = imageHits[iHit][0]
					img[iHit][1] = imageHits[iHit][1]
					img[iHit][2] = imageHits[iHit][2]
					img[iHit][3] = imageHits[iHit][3]
				bkg_imgs.append(np.concatenate(([fileNum, IDb],img.flatten())))
				bkg_infos.append(np.array([
					fileNum,
					IDb,
					0,
					nPV,
					track.deltaRToClosestElectron,
					track.deltaRToClosestMuon,
					track.deltaRToClosestTauHad,
					track_eta,
					track_phi,
					track.genMatchedID,
					track.genMatchedDR
				]))
				IDb+=1

np.savez_compressed("images"+fOut + '.npz', 
					e=e_imgs,
					bkg=bkg_imgs,
					e_infos=e_infos,
					bkg_infos=bkg_infos)