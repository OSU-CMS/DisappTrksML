#!/usr/bin/env python
from DisappTrksML.DeepSets.architecture import *

class MuonModel(DeepSetsArchitecture):
	
	def __init__(self, eta_range=0.25, phi_range=0.25, max_hits=100,
				 phi_layers=[64, 64, 256], f_layers=[64, 64, 64],
				 track_info_indices=[4, 6, 8, 9, 10, 11, 12, 13, 14, 15],
				 max_hits_calos = 100, phi_layers_calos= [64, 64, 256]):
		DeepSetsArchitecture.__init__(self, eta_range, phi_range, max_hits, phi_layers, f_layers, track_info_indices)
		self.input_shape = (self.max_hits, 7)
		self.max_hits_calos = max_hits_calos
		self.input_shape_calos = (self.max_hits_calos, 4)
		self.phi_layers_calos = phi_layers_calos
		self.track_info_shape = len(track_info_indices)

	def eventSelectionTraining(self, event):
		trackPasses = []
		for track in event.tracks:
			if (abs(track.eta) >= 2.4 or
				track.inGap or
				abs(track.dRMinJet) < 0.5 or
				abs(track.deltaRToClosestElectron) < 0.15 or
				abs(track.deltaRToClosestTauHad) < 0.15):
				trackPasses.append(False)
			else:
				trackPasses.append(True)
		return (True in trackPasses), trackPasses

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

				if isGenMatched(event, track, 13):
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

			print 'Wrote', outputFileName
		else:
			print 'No events found in file'

		inputFile.Close()

	def convertTPFileToNumpy(self, fileName):
		inputFile = TFile(fileName, 'read')
		inputTree = inputFile.Get('trackImageProducer/tree')

		tracks = []
		infos = []
		calos = []
		recoMuons = []
		recoMuons_infos = []

		for event in inputTree:
			eventPasses, trackPasses, trackPassesVeto = self.eventSelectionLeptonBackground(event, 'muons')
			if not eventPasses: continue

			for i, track in enumerate(event.tracks):
				if not trackPasses[i]: continue
				if not trackPassesVeto[i]: continue
				#if not track.isTagProbeMuon: continue

				values = self.convertTrackFromTree(event, track, 1)
				tracks.append(values['sets'])
				infos.append(values['infos'])
				# values = self.convertTrackFromTreeElectrons(event, track, 1)
				# calos.append(values['sets'])

		outputFileName = fileName.split('/')[-1] + '.npz'

		if len(tracks) > 0:
			np.savez_compressed(outputFileName,
								# recoMuons=recoMuons,
								# recoMuons_infos=recoMuons_infos,
								tracks=tracks,
								infos=infos)
								# signal_calos=calos)
			print 'Wrote', outputFileName
		else:
			print 'No events passed the selections'

		inputFile.Close()

	def convertAMSBFileToNumpy(self, fileName):
		inputFile = TFile(fileName, 'read')
		inputTree = inputFile.Get('trackImageProducer/tree')

		signal = []
		signal_infos = []
		signal_calos = []

		for event in inputTree:
			eventPasses, trackPasses = self.signalSelection(event)
			if not eventPasses: continue

			for i, track in enumerate(event.tracks):
				if not trackPasses[i]: continue

				if not (isGenMatched(event, track, 1000022) or isGenMatched(event, track, 1000024)): continue

				values = self.convertTrackFromTree(event, track, 1)
				signal.append(values['sets'])
				signal_infos.append(values['infos'])
				values = self.convertTrackFromTreeElectrons(event, track, 1)
				signal_calos.append(values['sets'])

		outputFileName = fileName.split('/')[-1] + '.npz'

		if len(signal) > 0:
			np.savez_compressed(outputFileName,
								signal=signal,
								signal_infos=signal_infos,
								signal_calos=signal_calos)
			print 'Wrote', outputFileName
		else:
			print 'No events passed the selections'

		inputFile.Close()

	def buildModel(self):
		inputs = Input(shape=(self.input_shape[-1],))

		# build phi network for each individual hit
		phi_network = Masking()(inputs)
		for layerSize in self.phi_layers[:-1]:
			phi_network = Dense(layerSize)(phi_network)
			phi_network = Activation('relu')(phi_network)
			#phi_network = BatchNormalization()(phi_network)
		phi_network = Dense(self.phi_layers[-1])(phi_network)
		phi_network = Activation('linear')(phi_network)

		# build summed model for latent space
		unsummed_model = Model(inputs=inputs, outputs=phi_network)
		set_input = Input(shape=self.input_shape)
		phi_set = TimeDistributed(unsummed_model)(set_input)
		summed = Lambda(lambda x: tf.reduce_sum(x, axis=1))(phi_set)
		phi_model = Model(inputs=set_input, outputs=summed)

		# define F (rho) network evaluating in the latent space
		if(self.track_info_shape == 0): f_inputs = Input(shape=(self.phi_layers[-1],)) # plus any other track/event-wide variable
		else: f_inputs = Input(shape=(self.phi_layers[-1]+self.track_info_shape,))
		f_network = Dense(self.f_layers[0])(f_inputs)
		f_network = Activation('relu')(f_network)
		for layerSize in self.f_layers[1:]:
			f_network = Dense(layerSize)(f_network)
			f_network = Activation('relu')(f_network)
		f_network = Dense(2)(f_network)
		f_outputs = Activation('softmax')(f_network)
		f_model = Model(inputs=f_inputs, outputs=f_outputs)

		# build the DeepSets architecture
		deepset_inputs = Input(shape=self.input_shape)
		latent_space = phi_model(deepset_inputs)
		if(self.track_info_shape == 0): 
			deepset_outputs = f_model(latent_space)
		else: 
			info_inputs = Input(shape=(self.track_info_shape,))
			deepset_inputs_withInfo = concatenate([latent_space,info_inputs])
			deepset_outputs = f_model(deepset_inputs_withInfo)

		if(self.track_info_shape == 0): model = Model(inputs=deepset_inputs, outputs=deepset_outputs)
		else: model = Model(inputs=[deepset_inputs,info_inputs], outputs=deepset_outputs)

		print(f_model.summary())
		print(phi_model.summary())
		print(model.summary())

		self.model = model

	def buildTwoHeadedModel(self):
		inputs = Input(shape=(self.input_shape[-1],))

		# build phi network for each individual hit
		phi_network = Masking()(inputs)
		for layerSize in self.phi_layers[:-1]:
			phi_network = Dense(layerSize)(phi_network)
			phi_network = Activation('relu')(phi_network)
			#phi_network = BatchNormalization()(phi_network)
		phi_network = Dense(self.phi_layers[-1])(phi_network)
		phi_network = Activation('linear')(phi_network)

		# build summed model for latent space
		unsummed_model = Model(inputs=inputs, outputs=phi_network)
		set_input = Input(shape=self.input_shape)
		phi_set = TimeDistributed(unsummed_model)(set_input)
		summed = Lambda(lambda x: tf.reduce_sum(x, axis=1))(phi_set)
		phi_model = Model(inputs=set_input, outputs=summed)


		# calo model
		inputs_calos = Input(shape=(self.input_shape_calos[-1],))

		# build phi network for each individual hit
		phi_network_calos = Masking()(inputs_calos)
		for layerSize in self.phi_layers_calos[:-1]:
			phi_network_calos = Dense(layerSize)(phi_network_calos)
			phi_network_calos = Activation('relu')(phi_network_calos)
			#phi_network = BatchNormalization()(phi_network)
		phi_networ_calosk = Dense(self.phi_layers_calos[-1])(phi_network_calos)
		phi_network_calos = Activation('linear')(phi_network_calos)

		# build summed model for latent space
		unsummed_model_calos = Model(inputs=inputs_calos, outputs=phi_network_calos)
		set_input_calos = Input(shape=self.input_shape_calos)
		phi_set_calos = TimeDistributed(unsummed_model_calos)(set_input_calos)
		summed_calos = Lambda(lambda x: tf.reduce_sum(x, axis=1))(phi_set_calos)
		phi_model_calos = Model(inputs=set_input_calos, outputs=summed_calos)

		# define F (rho) network evaluating in the latent space
		if(self.track_info_shape == 0): 
			f_inputs = Input(shape=(self.phi_layers[-1] + self.phi_layers_calos[-1],)) # plus any other track/event-wide variable
		else: 
			f_inputs = Input(shape=(self.phi_layers[-1]*2 + self.track_info_shape,))
		f_network = Dense(self.f_layers[0])(f_inputs)
		f_network = Activation('relu')(f_network)
		for layerSize in self.f_layers[1:]:
			f_network = Dense(layerSize)(f_network)
			f_network = Activation('relu')(f_network)
		f_network = Dense(2)(f_network)
		f_outputs = Activation('softmax')(f_network)
		f_model = Model(inputs=f_inputs, outputs=f_outputs)

		# build the DeepSets architecture
		deepset_inputs = Input(shape=self.input_shape)
		latent_space = phi_model(deepset_inputs)

		deepset_inputs_calos = Input(shape=self.input_shape_calos)
		latent_space_calos = phi_model(deepset_inputs_calos)

		if(self.track_info_shape == 0): 
			deepset_outputs = f_model(concatenate([latent_space, latent_space_calos]))
		else: 
			info_inputs = Input(shape=(self.track_info_shape,))
			deepset_inputs_withInfo = concatenate([latent_space,latent_space_calos,info_inputs])
			deepset_outputs = f_model(deepset_inputs_withInfo)

		if(self.track_info_shape == 0): model = Model(inputs=[deepset_inputs,deepset_inputs_calos], outputs=deepset_outputs)
		else: model = Model(inputs=[deepset_inputs,deepset_inputs_calos,info_inputs], outputs=deepset_outputs)

		print(f_model.summary())
		print(phi_model.summary())
		print(model.summary())

		self.model = model

	def evaluate_model(self, event, track):
		event = self.convertTrackFromTree(event, track, 1) # class_label doesn't matter
		prediction = self.model.predict([np.reshape(event['sets'], (1, self.max_hits, 4))[:, :self.max_hits, :], np.reshape(event['infos'], (1, len(event['infos'])))[:,self.track_info_indices]])
		return prediction[0, 1] # p(is electron)

	def evaluate_npy(self, fname, obj=['sets']):
		data = np.load(fname, allow_pickle=True)

		if(data[obj[0]].shape[0] == 0): return True, 0
		sets = data[obj[0]][:, :self.max_hits]

		x = [sets]

		if(len(obj) > 1):
			info = data[obj[1]][:, self.track_info_indices]
			x.append(info)
			
		return False, self.model.predict(x,)
