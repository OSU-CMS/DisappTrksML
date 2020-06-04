from __future__ import print_function

import sys
import math

from ROOT import TFile, TTree
import numpy as np

maxEta = 0.5
maxPhi = 0.5
nPixels = 50

if len(sys.argv) < 2:
	print('Usage: python convertData.py FILE')
	sys.exit(-1)

inputFile = TFile(sys.argv[1], 'read')
tree = inputFile.Get('trackImageProducer/tree')

nEvents = tree.GetEntries()

nElectrons = nMuons = nTaus = 0
for event in tree:
	for iTrack in range(len(event.track_eta)):
		if abs(event.track_genMatchedDR[iTrack]) > 0.1 or event.track_genMatchedPt[iTrack] < 10.0:
			continue

		if abs(event.track_genMatchedID[iTrack]) == 11:
			nElectrons += 1
		elif abs(event.track_genMatchedID[iTrack]) == 13:
			nMuons += 1
		elif abs(event.track_genMatchedID[iTrack]) == 15:
			nTaus += 1

# 4 dimensions: [track index, ix (eta) pixel, iy (phi) pixel, detector index] = hit energy/multiplicity
data_electrons = np.zeros((nElectrons, nPixels, nPixels, 3), dtype=float)
data_muons = np.zeros((nMuons, nPixels, nPixels, 3), dtype=float)
data_taus = np.zeros((nTaus, nPixels, nPixels, 3), dtype=float)

# 1 dimension: [track index] = bool(deltaRToClosestX < 0.15)
recoClass_electrons = np.zeros((nElectrons))
recoClass_muons = np.zeros((nMuons))
recoClass_taus = np.zeros((nTaus))

electronIndex = muonIndex = tauIndex = -1

# fill data
for iEvent, event in enumerate(tree):
	if iEvent % 1000 == 0:
		print('\t', iEvent, '/', nEvents, '...')

	# for tracks in event
	for iTrack in range(len(event.track_eta)):

		if abs(event.track_genMatchedDR[iTrack]) > 0.1 or event.track_genMatchedPt[iTrack] < 10.0:
			continue

		isEle  = (abs(event.track_genMatchedID[iTrack]) == 11)
		isMuon = (abs(event.track_genMatchedID[iTrack]) == 13)
		isTau  = (abs(event.track_genMatchedID[iTrack]) == 15)

		if isEle:
			electronIndex += 1
		elif isMuon: 
			muonIndex += 1
		elif isTau:
			tauIndex += 1
		else:
			continue

		for ix in range(nPixels):
			for iy in range(nPixels):
				if isEle:
					recoClass_electrons[electronIndex] = int(abs(event.track_deltaRToClosestElectron[iTrack]) < 0.15)
				elif isMuon:
					recoClass_muons[muonIndex] = int(abs(event.track_deltaRToClosestMuon[iTrack]) < 0.15)
				elif isTau:
					recoClass_taus[tauIndex] = int(abs(event.track_deltaRToClosestTauHad[iTrack]) < 0.15)

		# for recHits in event
		for iHit in range(len(event.recHits_eta)):

			# shift origin to track's eta/phi
			dEta = event.track_eta[iTrack] - event.recHits_eta[iHit]
			dPhi = event.track_phi[iTrack] - event.recHits_phi[iHit]
			if abs(dPhi) > math.pi:
				dPhi -= round(dPhi / (2. * math.pi)) * 2. * math.pi

			# apply window cut
			if abs(dEta) >= maxEta or abs(dPhi) >= maxPhi:
				continue

			# merge sub-subdetectors (EB, EE etc) into subdetectors (ECAL=0, HCAL=1, Muons=2)
			if event.recHits_detType[iHit] == 1 or event.recHits_detType[iHit] == 2:
				detIndex = 0
			elif event.recHits_detType[iHit] == 4:
				detIndex = 1
			elif event.recHits_detType[iHit] > 4:
				detIndex = 2
			else:
				continue

			# pixelize
			ix = int((dEta + maxEta) * nPixels / (2. * maxEta))
			iy = int((dPhi + maxPhi) * nPixels / (2. * maxPhi))
		
			# Use hit energy for ECAL/HCAL intensities, hit multiplicity for muons
			value = event.recHits_energy[iHit] if detIndex != 2 else 1

			if isEle:
				data_electrons[electronIndex, ix, iy, detIndex] += value
			elif isMuon:
				data_muons[muonIndex, ix, iy, detIndex] += value
			elif isTau:
				data_taus[tauIndex, ix, iy, detIndex] += value


np.save('electrons_' + sys.argv[1], data_electrons)
np.save('muons_' + sys.argv[1], data_muons)
np.save('taus_' + sys.argv[1], data_taus)

np.save('electronClasses_' + sys.argv[1], recoClass_electrons)
np.save('muonClasses_' + sys.argv[1], recoClass_muons)
np.save('tauClasses_' + sys.argv[1], recoClass_taus)

print('Converted --')
print('Electrons:', nElectrons)
print('Muons:', nMuons)
print('Taus:', nTaus)
