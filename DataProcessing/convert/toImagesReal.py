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
tanh_scaling = True
res_eta = 40
res_phi = 40
eta_ub,eta_lb = 0.25,-0.25
phi_ub,phi_lb = 0.25,-0.25
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

def convert_eta(eta):
	return int(round(((res_eta-1)*1.0/(eta_ub-eta_lb))*(eta-eta_lb)))

def convert_phi(phi):
	return int(round(((res_phi-1)*1.0/(phi_ub-phi_lb))*(phi-phi_lb)))

# combine EB+EE and muon detectors into ECAL/HCAL/MUO indices
def type_to_channel(hittype):
    #none
    if(hittype == 0): return -1
    #ECAL (EE,EB)
    if(hittype == 1 or hittype == 2): return 0
    #ES
    if(hittype == 3): return 1 
    #HCAL
    if(hittype == 4): return 2
    #Muon (CSC,DT,RPC)
    if(hittype == 5 or hittype ==6 or hittype == 7): return 3
    
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
#eRecoTree = fin.Get('eRecoTree')
bTree = fin.Get('bTree')
electrons, e_infos = [], []
bkg, bkg_infos = [], []
IDe, IDb = 0, 0

for class_label,tree in zip([0,1],[bTree,eTree]):
	ID = 0

	for event in tree:
		nPV = event.nPV
		eventNumber = event.eventNumber
		lumiBlockNumber = event.lumiBlockNumber
		runNumber = event.runNumber
		#eventNumber = -1
		#lumiBlockNumber = -1
		#runNumber = -1

		for track in event.tracks:
			
			matrix = np.zeros([res_eta,res_phi,4])

			momentum = XYZVector(track.px,track.py,track.pz)
			track_eta = momentum.Eta()
			track_phi = momentum.Phi()

			hit_energy = 0
			max_hit = [-1, -10, -10]       
			for iHit,hit in enumerate(event.recHits):

				if(hit.detType != 3): 
					hit_energy += hit.energy
				if(hit.energy > max_hit[0]):
					max_hit[0] = hit.energy
					max_hit[1] = hit.eta
					max_hit[2] = hit.phi

				dEta = track_eta - hit.eta
				dPhi = track_phi - hit.phi

				# branch cut [-pi, pi)
				if abs(dPhi) > math.pi:
					dPhi -= round(dPhi / (2. * math.pi)) * 2. * math.pi

				if(dPhi > phi_ub or dPhi < phi_lb): continue
				if(dEta > eta_ub or dEta < eta_lb): continue

				dEta = convert_eta(dEta)
				dPhi = convert_phi(dPhi)

				channel = type_to_channel(hit.detType)
				if(channel == -1): continue

				if channel != 3: matrix[dEta,dPhi,channel] += hit.energy
				else: matrix[dEta][dPhi][channel] += 1

			track_DR = math.sqrt((track_eta-max_hit[1])**2 + (track_phi-max_hit[2])**2) 
			
			# scaling options
			if(tanh_scaling): matrix = np.tanh(matrix)

			matrix = matrix.flatten().reshape([matrix.shape[0]*matrix.shape[1]*matrix.shape[2],])  
			matrix = matrix.astype('float32')
			matrix = np.concatenate(([fileNum, ID, eventNumber, lumiBlockNumber, runNumber],matrix))

			infos = np.array([fileNum, ID,
					eventNumber, lumiBlockNumber, runNumber,
					class_label,
					nPV,
					track.deltaRToClosestElectron,
					track.deltaRToClosestMuon,
					track.deltaRToClosestTauHad,
					track_eta,
					track_phi,
					track.dRMinBadEcalChannel,
					track_DR,
					hit_energy,
					max_hit[0]	
				])

			if(class_label == 0):
				electrons.append(matrix)
				e_infos.append(infos)
				ID+=1
			if(class_label == 1):
				bkg.append(matrix)
				bkg_infos.append(infos)
				ID+=1

		
np.savez_compressed('images_b_'+str(fileNum)+'.npz', 
					images=electrons,
					infos=e_infos)
np.savez_compressed('images_e_'+str(fileNum)+'.npz', 
					images=bkg,
					infos=bkg_infos)
np.save('images_b_' + str(fileNum), electrons)
np.save("images_e_" + str(fileNum), bkg)

