#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')

import math
from datetime import datetime
import numpy as np
import pickle
import random

from ROOT import TFile, TTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn import preprocessing

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
	dEta = track.eta - hit.eta
	dPhi = track.phi - hit.phi
	# branch cut [-pi, pi)
	if abs(dPhi) > math.pi:
		dPhi -= round(dPhi / (2. * math.pi)) * 2. * math.pi
	return (dEta, dPhi)

def isGenMatched(event, track, pdgId):
	matchID = 0
	matchDR2 = -1
	for p in event.genParticles:
		if p.pt < 10:
			continue
		if not p.isPromptFinalState and not p.isDirectPromptTauDecayProductFinalState:
			continue

		dEta = track.eta - p.eta
		dPhi = track.phi - p.phi
		if abs(dPhi) > math.pi:
			dPhi -= round(dPhi / (2. * math.pi)) * 2. * math.pi
		dR2 = dEta*dEta + dPhi*dPhi

		if matchDR2 < 0 or dR2 < matchDR2:
			matchDR2 = dR2
			matchID = p.pdgId

	return (abs(matchID) == pdgId and abs(matchDR2) < 0.1**2)

# def printEventInfo(event, track):
# 	print 'EVENT INFO'
# 	print 'Trigger:', event.firesGrandOrTrigger
# 	print 'MET filters:', event.passMETFilters
# 	print 'Num good PVs (>=1):', event.numGoodPVs
# 	print 'MET (no mu) (>120):', event.metNoMu
# 	print 'Num good jets (>=1):', event.numGoodJets
# 	print 'max dijet dPhi (<=2.5):', event.dijetDeltaPhiMax
# 	print 'dPhi(lead jet, met no mu) (>0.5):', abs(event.leadingJetMetPhi)
# 	print
# 	print 'TRACK INFO'
# 	print '\teta (<2.1):', abs(track.eta)
# 	print '\tpt (>55):', track.pt
# 	print '\tIn gap (false):', track.inGap
# 	print '\tNot in 2017 low eff. region (true):', (track.phi < 2.7 or track.eta < 0 or track.eta > 1.42)
# 	print '\tmin dR(track, bad ecal channel) (>= 0.05):', track.dRMinBadEcalChannel
# 	print '\tnValidPixelHits (>=4):', track.nValidPixelHits
# 	print '\tnValidHits (>=4):', track.nValidHits
# 	print '\tmissing inner hits (==0):', track.missingInnerHits
# 	print '\tmissing middle hits (==0):', track.missingMiddleHits
# 	print '\ttrackIso/pt (<0.05):', track.trackIso / track.pt
# 	print '\td0 (<0.02):', abs(track.d0)
# 	print '\tdz (<0.5):', abs(track.dz)
# 	print '\tmin dR(track, jet) (>0.5):', abs(track.dRMinJet)
# 	print '\tmin dR(track, ele) (>0.15):', abs(track.deltaRToClosestElectron)
# 	print '\tmin dR(track, muon) (>0.15):', abs(track.deltaRToClosestMuon)
# 	print '\tmin dR(track, tauHad) (>0.15):', abs(track.deltaRToClosestTauHad)
# 	print '\tecalo (<10):', track.ecalo
# 	print '\tmissing outer hits (>=3):', track.missingOuterHits
# 	print
# 	print '\tisTagProbeElectron:', track.isTagProbeElectron
# 	print '\tisTagProbeMuon:', track.isTagProbeMuon

class GeneralArchitecture:

	def __init__(self, eta_range=0.25, phi_range=0.25, max_hits=100,model=None):
		self.eta_range = eta_range
		self.phi_range = phi_range
		self.max_hits = max_hits

		self.input_shape = (self.max_hits, 4)

		self.model = model

	def eventSelectionTraining(self, event):
		trackPasses = []
		for track in event.tracks:
			if (abs(track.eta) >= 2.4 or
				track.inGap or
				abs(track.dRMinJet) < 0.5 or
				abs(track.deltaRToClosestElectron) < 0.15 or
				#abs(track.deltaRToClosestMuon) < 0.15 or
				abs(track.deltaRToClosestTauHad) < 0.15):
				trackPasses.append(False)
			else:
				trackPasses.append(True)
		return (True in trackPasses), trackPasses

	def eventSelectionSignal(self, event):
		eventPasses = (event.firesGrandOrTrigger == 1 and
			event.passMETFilters == 1 and
			event.numGoodPVs >= 1 and
			event.metNoMu > 120 and
			event.numGoodJets >= 1 and
			event.dijetDeltaPhiMax <= 2.5 and
			abs(event.leadingJetMetPhi) > 0.5)

		trackPasses = [False] * len(event.tracks)

		if not eventPasses:
			return eventPasses, trackPasses

		for i, track in enumerate(event.tracks):
			if (not abs(track.eta) < 2.1 or
				not track.pt > 55 or
				track.inGap == 0 or
				not (track.phi < 2.7 or track.eta < 0 or track.eta > 1.42)): # 2017 eta-phi low efficiency
				continue

			if (not track.dRMinBadEcalChannel >= 0.05 or
				not track.nValidPixelHits >= 4 or
				not track.nValidHits >= 4 or
				not track.missingInnerHits == 0 or
				not track.missingMiddleHits == 0 or
				not track.trackIso / track.pt < 0.05 or
				not abs(track.d0) < 0.02 or
				not abs(track.dz) < 0.5 or
				not abs(track.dRMinJet) > 0.5 or
				not abs(track.deltaRToClosestElectron) > 0.15 or
				not abs(track.deltaRToClosestMuon) > 0.15 or
				not abs(track.deltaRToClosestTauHad) > 0.15 or
				not track.ecalo < 10 or
				not track.missingOuterHits >= 3):
				continue

			trackPasses[i] = True

		return (True in trackPasses), trackPasses

	def eventSelectionFakeBackground(self, event):
		eventPasses = (event.passMETFilters == 1)
		trackPasses = [False] * len(event.tracks)

		if not eventPasses:
			return eventPasses, trackPasses

		for i, track in enumerate(event.tracks):
			if (not abs(track.eta) < 2.1 or
				not track.pt > 30 or
				track.inGap == 0 or
				not (track.phi < 2.7 or track.eta < 0 or track.eta > 1.42)): # 2017 eta-phi low efficiency
				continue

			if (not track.dRMinBadEcalChannel >= 0.05 or
				not track.nValidPixelHits >= 4 or
				not track.nValidHits >= 4 or
				not track.missingInnerHits == 0 or
				not track.missingMiddleHits == 0 or
				not track.trackIso / track.pt < 0.05 or
				# d0 sideband
				not abs(track.d0) >= 0.05 or
				not abs(track.d0) < 0.5 or
				not abs(track.dz) < 0.5 or
				not abs(track.dRMinJet) > 0.5 or
				not abs(track.deltaRToClosestElectron) > 0.15 or
				not abs(track.deltaRToClosestMuon) > 0.15 or
				not abs(track.deltaRToClosestTauHad) > 0.15 or
				not track.ecalo < 10 or
				not track.missingOuterHits >= 3):
				continue

			trackPasses[i] = True

		return (True in trackPasses), trackPasses

	def eventSelectionLeptonBackground(self, event, lepton_type):

		eventPasses = (event.passMETFilters == 1)
		trackPasses = [False] * len(event.tracks)
		trackPassesVeto = [False] * len(event.tracks)

		if not eventPasses:
			return eventPasses, trackPasses, trackPassesVeto

		for i, track in enumerate(event.tracks):
			if (not abs(track.eta) < 2.1 or
				not track.pt > 30 or
				track.inGap == 0 or
				not (track.phi < 2.7 or track.eta < 0 or track.eta > 1.42)): # 2017 eta-phi low efficiency
				continue

			if (lepton_type == 'electrons' and not track.isTagProbeElectron == 1):
				continue
			if (lepton_type == 'muons' and not track.isTagProbeMuon == 1):
				continue

			if (not track.dRMinBadEcalChannel >= 0.05 or
				not track.nValidPixelHits >= 4 or
				not track.nValidHits >= 4 or
				not track.missingInnerHits == 0 or
				not track.missingMiddleHits == 0 or
				not track.trackIso / track.pt < 0.05 or
				not abs(track.d0) < 0.02 or
				not abs(track.dz) < 0.5 or
				not abs(track.dRMinJet) > 0.5 or
				not abs(track.deltaRToClosestTauHad) > 0.15):
				continue

			if (lepton_type == 'electrons' and not abs(track.deltaRToClosestMuon) > 0.15):
				continue
			if (lepton_type == 'muons' and (not abs(track.deltaRToClosestElectron) > 0.15 or not track.ecalo < 10)):
				continue

			trackPasses[i] = True

			if lepton_type == 'electrons':
				if (abs(track.deltaRToClosestElectron) > 0.15 and
					track.ecalo < 10 and
					track.missingOuterHits >= 3):
					trackPassesVeto[i] = True
			if lepton_type == 'muons':
				if (abs(track.deltaRToClosestMuon) > 0.15 and
					track.missingOuterHits >= 3):
					trackPassesVeto[i] = True

		return (True in trackPasses), trackPasses, trackPassesVeto

	def convertTrackFromTree(self, event, track, class_label):
		hits = []
		dists = []
		hcal_energy, ecal_energy = [], []

		for hit in event.recHits:
			dEta, dPhi = imageCoordinates(track, hit)

			if abs(dEta) >= self.eta_range or abs(dPhi) >= self.phi_range: continue

			# CSC
			if hit.detType == 5:
				station = hit.cscRecHits[0].station
				time = hit.time
				detTypeEncoded = [1,0,0]

			# DT
			elif hit.detType == 6:
				station = hit.dtRecHits[0].station
				time = hit.time
				detTypeEncoded = [0,1,0]

			# RPC
			elif hit.detType == 7:
				station = 0
				time = hit.time
				detTypeEncoded = [0,0,1]

			else: 
				if hit.detType == 4: hcal_energy.append(hit.energy)
				elif hit.detType == 1 or hit.detType == 2: ecal_energy.append(hit.energy)
				continue
			
			hits.append([dEta, dPhi, station, time] + detTypeEncoded)
			dists.append(dEta**2 + dPhi**2)

		# sort by closest hits to track in eta, phi
		if len(hits) > 0:
			hits = np.reshape(hits, (len(hits), 7))
			hits = hits[np.array(dists).argsort()]

		sets = np.zeros(self.input_shape)
		for i in range(min(len(hits), self.max_hits)):
			for j in range(7):
				sets[i][j] = hits[i][j]

		infos = np.array([event.eventNumber, event.lumiBlockNumber, event.runNumber,
						  class_label,
						  event.nPV,
						  track.deltaRToClosestElectron,
						  track.deltaRToClosestMuon,
						  track.deltaRToClosestTauHad,
						  track.eta,
						  track.phi,
						  track.dRMinBadEcalChannel,
						  track.nLayersWithMeasurement,
						  track.nValidPixelHits,
						  np.sum(ecal_energy),
						  np.sum(hcal_energy),
						  track.pt,
						  track.ptError,
						  track.normalizedChi2,
						  track.dEdxPixel,
						  track.dEdxStrip,
						  track.d0,
						  track.dz])

		values = {
			'sets' : sets,
			'infos' : infos,
		}

		return values

	def convertMCFileToNumpy(self, fileName):
		inputFile = TFile(fileName, 'read')
		inputTree = inputFile.Get('trackImageProducer/tree')

		signal = []
		signal_info = []
		background = []
		background_info = []
		signal_calos = []
		background_calos = []

		for event in inputTree:
			eventPasses, trackPasses = self.eventSelectionTraining(event)
			if not eventPasses: continue

			for i, track in enumerate(event.tracks):
				if not trackPasses[i]: continue

				# gen matched, non reconsutrcted
				if isGenMatched(event, track, 13) and (abs(track.deltaRToClosestMuon) >= 0.15):
					values = self.convertTrackFromTree(event, track, 1)
					signal.append(values['sets'])
					signal_info.append(values['infos'])
					# values = self.convertTrackFromTreeElectrons(event, track, 1)
					# signal_calos.append(values['sets'])

		outputFileName = fileName.split('/')[-1] + '.npz'

		if len(signal) != 0 or len(background) != 0:

			np.savez_compressed(outputFileName,
								signal=signal,
								signal_info=signal_info,
								background=background,
								background_info=background_info)
								# signal_calos = signal_calos,
								# background_calos = background_calos)

			print('Wrote', outputFileName)
		else:
			print('No events found in file')

		inputFile.Close()

	def isProbeTrack(self, track):
		if(track.pt <= 30 or
		abs(track.eta) >= 2.1 or
		track.nValidPixelHits < 4 or
		track.nValidHits < 4 or
		track.missingInnerHits != 0 or
		track.missingMiddleHits != 0 or
		track.trackIso / track.pt >= 0.05 or
		abs(track.d0) >= 0.02 or
		abs(track.dz) >= 0.5 or
		abs(track.dRMinJet) <= 0.5):
			return False
		return True

	def convertTPFileToNumpy(self, fileName):
		inputFile = TFile(fileName, 'read')
		inputTree = inputFile.Get('trackImageProducer/tree')

		signal_tracks, signal_infos = [], []
		bkg_tracks, bkg_infos = [], []

		for event in inputTree:

			eventPasses, trackPasses = self.eventSelectionTraining(event)
			if not eventPasses: continue

			for i, track in enumerate(event.tracks):
				if abs(track.deltaRToClosestMuon) < 0.15: continue
				if not trackPasses[i]: continue

				if track.isTagProbeMuon:
					values = self.convertTrackFromTree(event, track, 1)
					signal_tracks.append(values['sets'])
					signal_infos.append(values['infos'])

				if (not track.isTagProbeMuon) and self.isProbeTrack(track):
					values = self.convertTrackFromTree(event, track, 0)
					bkg_tracks.append(values['sets'])
					bkg_infos.append(values['infos'])

		outputFileName = fileName.split('/')[-1] + '.npz'

		if len(signal_tracks) > 0 or len(bkg_tracks) > 0:
			np.savez_compressed(outputFileName,
								signal_tracks = signal_tracks,
								signal_infos = signal_infos,
								bkg_tracks = bkg_tracks,
								bkg_infos = bkg_infos)
			print('Wrote', outputFileName)
		else:
			print('No events passed the selections')

		inputFile.Close()

	def convertAMSBFileToNumpy(self, fileName):
		inputFile = TFile(fileName, 'read')
		inputTree = inputFile.Get('trackImageProducer/tree')

		signal = []
		signal_infos = []

		for event in inputTree:
			eventPasses, trackPasses = self.eventSelectionSignal(event)
			if not eventPasses: continue

			for i, track in enumerate(event.tracks):
				if not trackPasses[i]: continue

				if not (isGenMatched(event, track, 1000022) or isGenMatched(event, track, 1000024)): continue

				values = self.convertTrackFromTree(event, track, 1)
				signal.append(values['sets'])
				signal_infos.append(values['infos'])

		outputFileName = fileName.split('/')[-1] + '.npz'

		if len(signal) > 0:
			np.savez_compressed(outputFileName,
								signal=signal,
								signal_infos=signal_infos)
			print('Wrote', outputFileName)
		else:
			print('No events passed the selections')

		inputFile.Close()


	# plot precision and recall for different classifier outputs
	def metrics_per_cut(self, true, preds, nsplits=20):
		precisions, recalls, f1s, splits = [],[],[], []

		for split in np.arange(0,1,1.0/nsplits):
			class_labels = np.zeros(len(preds),dtype='int')
			class_labels[np.where(preds > split)] = 1
   
			cm = self.calc_cm(true,class_labels)

			precision, recall, f1 = self.calc_binary_metrics(cm)
			precisions.append(precision)
			recalls.append(recall)
			f1s.append(f1)
			splits.append(split)

		metrics = {
			'splits':splits,
			'precision':precisions,
			'recall':recalls,
			'f1':f1s
		}

		return metrics

	# load a list of files
	def load_from_files(self, files, dirname='', obj='tracks'):
		X = None
		for file in files:
			infile = np.load(dirname + file, allow_pickle=True)[obj]
			if len(infile) == 0: continue
			if X is None: X = infile
			else: X = np.vstack((X, infile))
		return X

	# calculate confusion matrix
	def calc_cm(self, true, predictions, dim=2):
		confusion_matrix = np.zeros((dim, dim))
		for t,p in zip(true.astype(int), predictions.astype(int)):
			confusion_matrix[t,p] += 1
		return confusion_matrix

	# calculate binary confusion matrix
	def calc_binary_cm(self, true, predictions, cutoff=0.5):
		predictions[predictions >= cutoff] = 1
		predictions[predictions < cutoff] = 0
		confusion_matrix = np.zeros((2, 2))
		for t,p in zip(true.astype(int), predictions.astype(int)):
			confusion_matrix[t,p] += 1
		return confusion_matrix

	# expects cm[c1,c1] = TP, cm[c2,c2] = TN
	def calc_binary_metrics(self, confusion_matrix, c1=1, c2=0):
		TP = confusion_matrix[c1][c1]
		FP = confusion_matrix[c2][c1]
		FN = confusion_matrix[c1][c2]
		TN = confusion_matrix[c2][c2]

		if((TP+FP) == 0): precision = 0
		else: precision = TP / (TP + FP)
		if((TP+FN) == 0): recall = 0
		else: recall = TP / (TP + FN)

		f1 = TP / (TP + 0.5*(FP + FN))

		return precision, recall, f1
