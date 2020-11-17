#!/usr/bin/env python

import os
import sys

from ROOT import TChain, TFile, TTree, TH1D, TH2D

from tensorflow.keras.layers import concatenate

from DisappTrksML.DeepSets.architecture import *

arch = DeepSetsArchitecture()
arch.buildModel()
arch.load_model_weights('noReco_64.64.256_64.64.64.h5')

def calculateFidicualMaps(electron_payload, muon_payload, payload_suffix):

	fiducial_maps = {}

	for flavor in ['ele', 'muon']:
		fin = TFile(electron_payload if flavor is 'ele' else muon_payload, 'read')
		beforeVeto = fin.Get('beforeVeto' + payload_suffix)
		afterVeto = fin.Get('afterVeto' + payload_suffix)

		totalEventsBeforeVeto = 0
		totalEventsAfterVeto = 0
		nRegionsWithTag = 0

		# loop over all bins in eta-phi and count events before/after to calculate the mean inefficiency
		for xbin in range(1, beforeVeto.GetXaxis().GetNbins()):
			for ybin in range(1, beforeVeto.GetYaxis().GetNbins()):
				if beforeVeto.GetBinContent(xbin, ybin) > 0:
					nRegionsWithTag += 1
				else:
					continue
				totalEventsBeforeVeto += beforeVeto.GetBinContent(xbin, ybin)
				totalEventsAfterVeto  += afterVeto.GetBinContent(xbin, ybin)
		meanInefficiency = totalEventsAfterVeto / totalEventsBeforeVeto

		# now with the mean, calculate the standard deviation as stdDev^2 = 1/(N-1) * sum(inefficiency - meanInefficiency)^2
		stdDevInefficiency = 0

		efficiency = afterVeto.Clone(flavor + '_efficiency')
		efficiency.SetDirectory(0)
		efficiency.Divide(beforeVeto)

		for xbin in range(1, efficiency.GetXaxis().GetNbins()):
			for ybin in range(1, efficiency.GetYaxis().GetNbins()):
				if beforeVeto.GetBinContent(xbin, ybin) == 0:
					continue
				thisInefficiency = efficiency.GetBinContent(xbin, ybin)
				stdDevInefficiency += (thisInefficiency - meanInefficiency)**2

		if nRegionsWithTag < 2:
			print 'Only ', nRegionsWithTag, ' regions with a tag lepton exist, cannot calculate fiducial map!!!'
			return hotSpots

		stdDevInefficiency /= nRegionsWithTag - 1
		stdDevInefficiency = math.sqrt(stdDevInefficiency)

		efficiency.Scale(1. / stdDevInefficiency)

		fiducial_maps[flavor] = efficiency

	return fiducial_maps

def fiducialMapSigmas(track, fiducial_maps):
	iBin = fiducial_maps['ele'].GetXaxis().FindBin(track.eta)
	jBin = fiducial_maps['ele'].GetYaxis().FindBin(track.phi)
	
	ele_value  = fiducial_maps['ele'].GetBinContent(iBin, jBin)
	muon_value = fiducial_maps['muon'].GetBinContent(iBin, jBin)

	return (ele_value, muon_value)

def signalSelection(event, fiducial_maps, fiducial_map_cut=-1):

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

		if fiducial_map_cut > 0:
			sigmas = fiducialMapSigmas(track, fiducial_maps)
			if sigmas[0] > fiducial_map_cut or sigmas[1] > fiducial_map_cut: 
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

def fakeBackgroundSelection(event, fiducial_maps, fiducial_map_cut=-1):
	eventPasses = (event.passMETFilters == 1)
	trackPasses = [False] * len(event.tracks)

	if not eventPasses:
		return eventPasses, trackPasses, trackPassesVeto

	for i, track in enumerate(event.tracks):
		if (not abs(track.eta) < 2.1 or
			not track.pt > 30 or
			track.inGap == 0 or
			not (track.phi < 2.7 or track.eta < 0 or track.eta > 1.42)): # 2017 eta-phi low efficiency
			continue

		if fiducial_map_cut > 0:
			sigmas = fiducialMapSigmas(track, fiducial_maps)
			if sigmas[0] > fiducial_map_cut or sigmas[1] > fiducial_map_cut: 
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

def leptonBackgroundSelection(event, fiducial_maps, lepton_type, fiducial_map_cut=-1):

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

		if fiducial_map_cut > 0:
			sigmas = fiducialMapSigmas(track, fiducial_maps)
			if sigmas[0] > fiducial_map_cut or sigmas[1] > fiducial_map_cut: 
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

def printEventInfo(event, track):
	print 'EVENT INFO'
	print 'Trigger:', event.firesGrandOrTrigger
	print 'MET filters:', event.passMETFilters
	print 'Num good PVs (>=1):', event.numGoodPVs
	print 'MET (no mu) (>120):', event.metNoMu
	print 'Num good jets (>=1):', event.numGoodJets
	print 'max dijet dPhi (<=2.5):', event.dijetDeltaPhiMax
	print 'dPhi(lead jet, met no mu) (>0.5):', abs(event.leadingJetMetPhi)
	print
	print 'TRACK INFO'
	print '\teta (<2.1):', abs(track.eta)
	print '\tpt (>55):', track.pt
	print '\tIn gap (false):', track.inGap
	print '\tNot in 2017 low eff. region (true):', (track.phi < 2.7 or track.eta < 0 or track.eta > 1.42)
	print '\tmin dR(track, bad ecal channel) (>= 0.05):', track.dRMinBadEcalChannel
	print '\tnValidPixelHits (>=4):', track.nValidPixelHits
	print '\tnValidHits (>=4):', track.nValidHits
	print '\tmissing inner hits (==0):', track.missingInnerHits
	print '\tmissing middle hits (==0):', track.missingMiddleHits
	print '\ttrackIso/pt (<0.05):', track.trackIso / track.pt
	print '\td0 (<0.02):', abs(track.d0)
	print '\tdz (<0.5):', abs(track.dz)
	print '\tmin dR(track, jet) (>0.5):', abs(track.dRMinJet)
	print '\tmin dR(track, ele) (>0.15):', abs(track.deltaRToClosestElectron)
	print '\tmin dR(track, muon) (>0.15):', abs(track.deltaRToClosestMuon)
	print '\tmin dR(track, tauHad) (>0.15):', abs(track.deltaRToClosestTauHad)
	print '\tecalo (<10):', track.ecalo
	print '\tmissing outer hits (>=3):', track.missingOuterHits
	print
	print '\tisTagProbeElectron:', track.isTagProbeElectron
	print '\tisTagProbeMuon:', track.isTagProbeMuon

def processDataset(dataset, inputDir):
	payload_dir = os.environ['CMSSW_BASE'] + '/src/OSUT3Analysis/Configuration/data/'
	fiducial_maps = calculateFidicualMaps(payload_dir + 'electronFiducialMap_2017_data.root', 
										  payload_dir + 'muonFiducialMap_2017_data.root', '_2017F')

	h_disc  = TH2D('disc', 'disc:discriminant:nLayers', 100, 0, 1, 10, 0, 10)
	h_sigma = TH2D('sigma', 'sigma:sigma:nLayers', 100, 0, 10, 10, 0, 10)
	h_disc_vs_sigma = TH2D('disc_vs_sigma', 'disc_vs_sigma:sigma:discriminant', 100, 0, 10, 100, 0, 1)

	chain = TChain('trackImageProducer/tree')
	chain.Add(inputDir)

	nEvents = chain.GetEntries()
	print '\nAdded', nEvents, 'events for dataset type:', dataset, '\n'

	firstTrack = False

	for iEvent, event in enumerate(chain):
		if iEvent % 10000 == 0:
			print '\tEvent', iEvent, '/', nEvents, '...'

		if dataset.startswith('higgsino'):
			eventPasses, trackPasses = signalSelection(event, fiducial_maps)
		elif dataset == 'fake':
			eventPasses, trackPasses = fakeBackgroundSelection(event, fiducial_maps)
		else:
			eventPasses, trackPasses, trackPassesVeto = leptonBackgroundSelection(event, fiducial_maps, dataset)

		if not eventPasses: continue

		for iTrack, track in enumerate(event.tracks):
			if not trackPasses[iTrack]: continue
			if dataset in ['electrons', 'muons'] and not trackPassesVeto[iTrack]: continue

			if not firstTrack:
				printEventInfo(event, track)
				firstTrack = True

			discriminant = arch.evaluate_model(event, track)
			fiducial_sigmas = fiducialMapSigmas(track, fiducial_maps)

			h_disc.Fill(discriminant, track.nLayersWithMeasurement)
			h_sigma.Fill(max(fiducial_sigmas), track.nLayersWithMeasurement)
			h_disc_vs_sigma.Fill(max(fiducial_sigmas), discriminant)

	outputFile = TFile('output_' + dataset + '.root', 'recreate')
	h_disc.Write()
	h_sigma.Write()
	h_disc_vs_sigma.Write()
	outputFile.Close()

def analyze():
	input_ele = TFile('output_electrons.root', 'read')
	input_muon = TFile('output_muons.root', 'read')
	input_fake = TFile('output_fake.root', 'read')
	input_signal = TFile('output_higgsino_300_1000.root', 'read')

	h_disc_ele = input_ele.Get('disc')
	h_disc_muon = input_muon.Get('disc')
	h_disc_fake = input_fake.Get('disc')
	h_disc_higgsino_300_1000 = input_signal.Get('disc')

	histos = []

	for i in [4, 5, 6]:
		ibin = h_disc_ele.GetYaxis().FindBin(i)
		histos.append(h_disc_ele.ProjectionX ('bkg_' + str(i) + '_ele',  ibin, ibin if i < 6 else -1))
		histos.append(h_disc_muon.ProjectionX('bkg_' + str(i) + '_mu',   ibin, ibin if i < 6 else -1))
		histos.append(h_disc_fake.ProjectionX('bkg_' + str(i) + '_fake', ibin, ibin if i < 6 else -1))
		histos.append(h_disc_higgsino_300_1000.ProjectionX('bkg_' + str(i) + '_higgsino_300_1000', ibin, ibin if i < 6 else -1))

	fout = TFile('durp.root', 'recreate')
	for histo in histos:
		histo.Write()
	fout.Close()

#######################################

datasets = {
	'electrons' : '/store/user/bfrancis/images_v5/SingleEle_2017F/hist_*.root',
	'muons'     : '/store/user/bfrancis/images_v5/SingleMu_2017F/hist_*.root',
	'fake'      : '/store/user/bfrancis/images_v5/ZtoMuMu_2017F/hist_*.root',
	'higgsino_300_1000' : '/data/users/bfrancis/condor/2017/images_higgsino_300_1000/hist_*.root',
}

for dataset in datasets:
	processDataset(dataset, datasets[dataset])

analyze()


