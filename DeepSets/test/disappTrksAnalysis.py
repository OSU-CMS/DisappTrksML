#!/usr/bin/env python

import os
import sys

from ROOT import TChain, TFile, TTree

from tensorflow.keras.layers import concatenate

from DisappTrksML.DeepSets.architecture import *

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
	iBin = electron_map.GetXaxis().FindBin(track.eta)
	jBin = electron_map.GetYaxis().FindBin(track.phi)
	
	ele_value  = fiducial_maps['ele'].GetBinContent(iBin, jBin)
	muon_value = fiducial_maps['muon'].GetBinContent(iBin, jBin)

	return (ele_value, muon_value)

def eventSelection(event, fiducial_maps, fiducial_map_cut=-1):
	trackPasses = []

	if (not event.firesGrandOrTrigger or
		not event.passMETFilters or
		not event.numGoodPVs >= 1 or
		not event.metNoMu > 120 or
		not event.numGoodJets >= 1 or
		not event.dijetDeltaPhiMax <= 2.5 or
		not abs(event.leadingJetMetPhi) > 0.5):
		return False, trackPasses

	for track in event.tracks:
		if (not abs(track.eta) < 2.1 or
			not track.pt > 55 or
			track.inGap or
			not (track.phi < 2.7 or track.eta < 0 or track.eta > 1.42)): # 2017 eta-phi low efficiency
			trackPasses.append(False)
			continue

		if fiducial_map_cut > 0:
			sigmas = fiducialMapSigmas(track, fiducial_maps)
			if sigmas[0] > fiducial_map_cut or sigmas[1] > fiducial_map_cut: 
				trackPasses.append(False)
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
			trackPasses.append(False)
			continue

		trackPasses.append(True)

	return (True in trackPasses), trackPasses

def buildModelWithEventInfo(input_shape=(100,4), info_shape=5, phi_layers=[64, 64, 256], f_layers=[64, 64, 64]):
	inputs = Input(shape=(input_shape[-1],))

	# build phi network for each individual hit
	phi_network = Masking()(inputs)
	for layerSize in phi_layers[:-1]:
		phi_network = Dense(layerSize)(phi_network)
		phi_network = Activation('relu')(phi_network)
		phi_network = BatchNormalization()(phi_network)
	phi_network = Dense(phi_layers[-1])(phi_network)
	phi_network = Activation('linear')(phi_network)

	# build summed model for latent space
	unsummed_model = Model(inputs=inputs, outputs=phi_network)
	set_input = Input(shape=input_shape)
	phi_set = TimeDistributed(unsummed_model)(set_input)
	summed = Lambda(lambda x: tf.reduce_sum(x, axis=1))(phi_set)
	phi_model = Model(inputs=set_input, outputs=summed)

	# define F (rho) network evaluating in the latent space
	f_inputs = Input(shape=(phi_layers[-1]+info_shape,)) # plus any other track/event-wide variable
	f_network = Dense(f_layers[0])(f_inputs)
	f_network = Activation('relu')(f_network)
	for layerSize in f_layers[1:]:
		f_network = Dense(layerSize)(f_network)
		f_network = Activation('relu')(f_network)
	f_network = Dense(2)(f_network)
	f_outputs = Activation('softmax')(f_network)
	f_model = Model(inputs=f_inputs, outputs=f_outputs)

	# build the DeepSets architecture
	deepset_inputs = Input(shape=input_shape)
	latent_space = phi_model(deepset_inputs)
	info_inputs = Input(shape=(info_shape,))
	deepset_inputs2 = concatenate([latent_space,info_inputs])
	deepset_outputs = f_model(deepset_inputs2)
	model = Model(inputs=[deepset_inputs,info_inputs], outputs=deepset_outputs)

	print(model.summary())

	model.compile(optimizer=keras.optimizers.Adam(), 
				  loss='categorical_crossentropy', 
				  metrics=['accuracy'])

	return model

def main():

	payload_dir = os.environ['CMSSW_BASE'] + '/src/OSUT3Analysis/Configuration/data/'
	fiducial_maps = calculateFidicualMaps(payload_dir + 'electronFiducialMap_2017_data.root', payload_dir + 'muonFiducialMap_2017_data.root', '')

	chain = TChain('trackImageProducer/tree')
	chain.Add('/store/user/bfrancis/images_v5/SingleMu_2017F/*.root')
	class_label = 1         # 1 for electons, 0 everything else

	nEvents = chain.GetEntries()
	print '\nAdded', nEvents, 'events\n'

	model = buildModelWithEventInfo()
	model.load_weights('model.5.h5')

	arch = DeepSetsArchitecture()

	for event in chain:

		eventPasses, trackPasses = eventSelection(event, fiducial_maps)
		if not eventPasses: continue

		for i, track in enumerate(event.tracks):
			if not trackPasses[i]: continue

			converted_arrays = arch.convertTrackFromTree(event, track, class_label)

			x = np.reshape(converted_arrays['sets'],(1,100,4))
			x_info = np.reshape(converted_arrays['infos'][[5,9,10,11,12]],(1,5))

			discriminant = model.predict([x,x_info])

			print(discriminant)
			sys.exit(0)

if __name__ == '__main__':

	main()
