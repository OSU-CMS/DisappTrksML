#!/usr/bin/env python
from DisappTrksML.DeepSets.architecture import *
import sys

class ElectronModel(DeepSetsArchitecture):

	def __init__(self,
				 eta_range=0.25, phi_range=0.25,
				 max_hits=100,
				 phi_layers=[64, 64, 256], f_layers =[64, 64, 64],
				 track_info_indices=[4, 8, 9, 12]):
		DeepSetsArchitecture.__init__(self, eta_range, phi_range, max_hits, phi_layers, f_layers, track_info_indices)
		self.track_info_shape = len(track_info_indices)

	def eventSelectionTraining(self, event):
		trackPasses = []
		for track in event.tracks:
			if (abs(track.eta) >= 2.4 or
				track.inGap or
				abs(track.dRMinJet) < 0.5 or
				abs(track.deltaRToClosestElectron) < 0.15 or
				abs(track.deltaRToClosestMuon) < 0.15 or
				abs(track.deltaRToClosestTauHad) < 0.15):
				trackPasses.append(False)
			else:
				trackPasses.append(True)
		return (True in trackPasses), trackPasses
		
	def convertTrackFromTree(self, event, track, class_label):
		hits = []
		miniHits = []

		for hit in event.recHits:
			dEta, dPhi = imageCoordinates(track, hit)
			if abs(dEta) >= self.eta_range or abs(dPhi) >= self.phi_range:
				continue
			detIndex = detectorIndex(hit.detType)
			energy = hit.energy if detIndex != 2 else 1
			hits.append((dEta, dPhi, energy, detIndex))

		for hit in event.miniRecHits:
			dEta, dPhi = imageCoordinates(track, hit)
			if abs(dEta) >= self.eta_range or abs(dPhi) >= self.phi_range:
				continue
			detIndex = detectorIndex(hit.detType)
			energy = hit.energy if detIndex != 2 else 1
			miniHits.append((dEta, dPhi, energy, detIndex))

		if len(hits) > 0:
			hits = np.reshape(hits, (len(hits), 4))
			hits = hits[hits[:, 2].argsort()]
			hits = np.flip(hits, axis=0)
			assert np.max(hits[:, 2]) == hits[0, 2]

		if len(miniHits) > 0:
			miniHits = np.reshape(miniHits, (len(miniHits), 4))
			miniHits = miniHits[miniHits[:, 2].argsort()]
			miniHits = np.flip(miniHits, axis=0)
			assert np.max(miniHits[:, 2]) == miniHits[0, 2]

		sets = np.zeros(self.input_shape)
		for i in range(min(len(hits), self.max_hits)):
			for j in range(4):
				sets[i][j] = hits[i][j]

		miniSets = np.zeros(self.input_shape)
		for i in range(min(len(miniHits), self.max_hits)):
			for j in range(4):
				miniSets[i][j] = miniHits[i][j]

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
						  track.nValidPixelHits])

		values = {
			'sets' : sets,
			'miniSets' : miniSets,
			'infos' : infos,
		}

		return values

	def convertMCFileToNumpy(self, fileName):
		inputFile = TFile(fileName, 'read')
		inputTree = inputFile.Get('trackImageProducer/tree')

		signal = []
		miniSignal = []
		signal_info = []
		background = []
		miniBackground = []
		background_info = []

		for event in inputTree:
			eventPasses, trackPasses = self.eventSelectionTraining(event)
			if not eventPasses: continue

			for i, track in enumerate(event.tracks):
				if not trackPasses[i]: continue
				
				if isGenMatched(event, track, 11):
					values = self.convertTrackFromTree(event, track, 1)
					signal.append(values['sets'])
					miniSignal.append(values['miniSets'])
					signal_info.append(values['infos'])
				else:
					values = self.convertTrackFromTree(event, track, 0)
					background.append(values['sets'])
					miniBackground.append(values['miniSets'])
					background_info.append(values['infos'])

		outputFileName = fileName.split('/')[-1] + '.npz'

		if len(signal) != 0 or len(background) != 0:
			np.savez_compressed(outputFileName,
								signal=signal,
								miniSignal=miniSignal,
								signal_info=signal_info,
								background=background,
								miniBackground=miniBackground,
								background_info=background_info)

			print('Wrote', outputFileName)
		else:
			print('No events found in file')

		inputFile.Close()

	def convertAMSBFileToNumpy(self, fileName, selection=None):
		inputFile = TFile(fileName, 'read')
		inputTree = inputFile.Get('trackImageProducer/tree')

		signal = []
		signal_infos = []

		for event in inputTree:
			if selection is None: sys.exit("Pick a selection to apply from ['full', 'training']")
			elif selection == 'full': eventPasses, trackPasses = self.eventSelectionSignal(event)
			elif selection == 'training': eventPasses, trackPasses = self.eventSelectionSignal(event)
			else: sys.exit("Selection not recognized.")
			
			if not eventPasses: continue

			for i, track in enumerate(event.tracks):
				if not trackPasses[i]: continue

				if not (isGenMatched(event, track, 1000022) or isGenMatched(event, track, 1000024)): continue

				values = self.convertTrackFromTree(event, track, 1)
				signal.append(values['sets'])
				signal_infos.append(values['infos'])

		outputFileName = (fileName.split('/')[-1]).split('.')[0] + '.npz'

		print(outputFileName)

		if len(signal) > 0:
			np.savez_compressed(outputFileName,
								signal=signal,
								signal_infos=signal_infos)
			print('Wrote', outputFileName)
		else:
			print('No events passed the selections')

		inputFile.Close()

	def convertTPFileToNumpy(self, fileName):
		inputFile = TFile(fileName, 'read')
		inputTree = inputFile.Get('trackImageProducer/tree')

		tracks = []
		infos = []

		for event in inputTree:
			eventPasses, trackPasses, trackPassesVeto = self.eventSelectionLeptonBackground(event, 'electrons')
			if not eventPasses: continue

			for i, track in enumerate(event.tracks):
				if not trackPasses[i]: continue
				if not trackPassesVeto[i]: continue

				values = self.convertTrackFromTree(event, track, 1)
				tracks.append(values['sets'])
				infos.append(values['infos'])	    

		if len(tracks) != 0:
			outputFileName = fileName.split('/')[-1] + '.npz'

			np.savez_compressed(outputFileName,
								tracks=tracks,
								infos=infos)

			print('Wrote', outputFileName)

			inputFile.Close()
		else:
			print('No events found in file')

	def buildModel(self):
		#Rewrite
		inputs = Input(shape = self.input_shape,  name = "input" )
		inputs_track = Input(shape = (self.track_info_shape,), name = "input_track" )
		print("input shape", self.input_shape, "input track shape", self.track_info_shape)
		phi_inputs = Input(shape = (self.input_shape[-1],))
		phi_network = Masking()(phi_inputs)
		for layerSize in self.phi_layers[:-1]:
			phi_network = Dense(layerSize)(phi_network)
			phi_network = Activation('relu')(phi_network)
			#phi_network = BatchNormalization()(phi_network)
		phi_network = Dense(self.phi_layers[-1])(phi_network)
		phi_network = Activation('linear')(phi_network)
		unsummed_model = Model(inputs=phi_inputs, outputs=phi_network)
		phi_set = TimeDistributed(unsummed_model)(inputs)
		summed = Lambda(lambda x: tf.reduce_sum(x, axis=1))(phi_set)
		if (self.track_info_shape == 0):
			f_network = Dense(self.f_layers[0])(summed)
		else:
			f_network = Dense(self.f_layers[0])(concatenate([summed,inputs_track]))
		f_network = Activation('relu')(f_network)
		for layerSize in self.f_layers[1:]:
			f_network = Dense(layerSize)(f_network)
			f_network = Activation('relu')(f_network)
		f_network = Dense(2)(f_network)
		f_outputs = Activation('softmax',name="output_xyz")(f_network)
		integratedmodel = Model(inputs=[inputs,inputs_track], outputs=f_outputs)

		print(integratedmodel.summary())

		self.model = integratedmodel

	def evaluate_model(self, event, track):
		event_converted = self.convertTrackFromTree(event, track, 1) # class_label doesn't matter
		prediction = self.model.predict([np.reshape(event_converted['sets'], (1, self.max_hits, 4)), np.reshape(event_converted['infos'], (1,len(event_converted['infos'])))[:, self.track_info_indices]])
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

	def saveGraph(self):
		cmsml.tensorflow.save_graph("graph.pb", self.model, variables_to_constants=True)
		cmsml.tensorflow.save_graph("graph.pb.txt", self.model, variables_to_constants=True)

