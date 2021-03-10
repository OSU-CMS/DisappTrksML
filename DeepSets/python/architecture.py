#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')

import math
from datetime import datetime
import numpy as np
import pickle

from ROOT import TFile, TTree

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, TimeDistributed, Masking, Input, Lambda, Activation, BatchNormalization, concatenate
from tensorflow.keras import optimizers, regularizers, callbacks

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

class DeepSetsArchitecture:

	model = None
	training_history = None

	def __init__(self, eta_range=0.25, phi_range=0.25, max_hits=100,
		phi_layers = [64, 64, 256], f_layers = [64, 64, 64], track_info_shape = 0):
		self.eta_range = eta_range
		self.phi_range = phi_range
		self.max_hits = max_hits

		self.input_shape = (self.max_hits, 4)
		self.track_info_shape = track_info_shape

		self.phi_layers = phi_layers
		self.f_layers = f_layers

	def convertTrackFromTreeElectrons(self, event, track, class_label):
		hits = []

		for hit in event.recHits:
			dEta, dPhi = imageCoordinates(track, hit)
			if abs(dEta) >= self.eta_range or abs(dPhi) >= self.phi_range:
				continue
			detIndex = detectorIndex(hit.detType)
			energy = hit.energy if detIndex != 2 else 1
			hits.append((dEta, dPhi, energy, detIndex))

		if len(hits) > 0:
			hits = np.reshape(hits, (len(hits), 4))
			hits = hits[hits[:, 2].argsort()]
			hits = np.flip(hits, axis=0)
			assert np.max(hits[:, 2]) == hits[0, 2]

		sets = np.zeros(self.input_shape)
		for i in range(min(len(hits), self.max_hits)):
			for j in range(4):
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
						  track.nValidPixelHits])

		values = {
			'sets' : sets,
			'infos' : infos,
		}

		return values

	def convertTrackFromTreeMuons(self, event, track, class_label):
		hits = []
		dists = []
		hcal_energy, ecal_energy = [], []

		for hit in event.recHits:

			dEta, dPhi = imageCoordinates(track, hit)
			if abs(dEta) >= self.eta_range or abs(dPhi) >= self.phi_range: continue

			# CSC
			if hit.detType == 5:
				station = hit.cscRecHits[0].station
				time = hit.cscRecHits[0].tpeak
			# DT
			elif hit.detType == 6:
				station = hit.dtRecHits[0].station
				time = hit.dtRecHits[0].digitime 

			# FIXME: add other detTypes
			else: 
				if hit.detType == 4: hcal_energy.append(hit.energy)
				elif hit.detType == 1 or hit.detType == 2: ecal_energy.append(hit.energy)
				continue
			
			hits.append((dEta, dPhi, station, time))
			dists.append(dEta**2 + dPhi**2)

		# sort by closest hits to track in eta, phi
		if len(hits) > 0:
			hits = np.reshape(hits, (len(hits), 4))
			hits = hits[np.array(dists).argsort()]

		sets = np.zeros(self.input_shape)
		for i in range(min(len(hits), self.max_hits)):
			for j in range(4):
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
						  ecal_energy,
						  hcal_energy])

		values = {
			'sets' : sets,
			'infos' : infos,
		}

		return values

	def eventSelection(self, event):
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

	def convertMCFileToNumpy(self, fileName):
		inputFile = TFile(fileName, 'read')
		inputTree = inputFile.Get('trackImageProducer/tree')

		signal = []
		signal_info = []
		background = []
		background_info = []

		for event in inputTree:
			eventPasses, trackPasses = self.eventSelection(event)
			if not eventPasses: continue

			for i, track in enumerate(event.tracks):
				if not trackPasses[i]: continue
				
				if isGenMatched(event, track, 13):
					values = self.convertTrackFromTreeMuons(event, track, 1)
					signal.append(values['sets'])
					signal_info.append(values['infos'])
				else:
					values = self.convertTrackFromTreeMuons(event, track, 0)
					background.append(values['sets'])
					background_info.append(values['infos'])

		# for event in inputTree:
		#     eventPasses, trackPasses = self.eventSelection(event)
		#     if not eventPasses: continue

		#     for i, track in enumerate(event.tracks):
		#         if not trackPasses[i]: continue
				
		#         # only gen-truth muons
		#         if not isGenMatched(event, track, 13): continue

		#         # non-reco muons
		#         if abs(track.deltaRToClosestMuon) >= 0.15 :
		#             values = self.convertTrackFromTreeMuons(event, track, 1)
		#             signal.append(values['sets'])
		#             signal_info.append(values['infos'])

		#         # reco muons
		#         else:
		#             values = self.convertTrackFromTreeMuons(event, track, 0)
		#             background.append(values['sets'])
		#             background_info.append(values['infos'])

		outputFileName = fileName.split('/')[-1] + '.npz'

		if len(signal) != 0 or len(background) != 0:
			np.savez_compressed(outputFileName,
								signal=signal,
								signal_info=signal_info,
								background=background,
								background_info=background_info)

			print 'Wrote', outputFileName
		else:
			print 'No events found in file'

		inputFile.Close()

	def convertFileToNumpy(self, fileName):
		inputFile = TFile(fileName, 'read')
		inputTree = inputFile.Get('trackImageProducer/tree')

		tracks = []
		infos = []

		for event in inputTree:
			eventPasses, trackPasses = self.eventSelection(event)
			if not eventPasses: continue

			for i, track in enumerate(event.tracks):
				if not trackPasses[i]: continue

				values = self.convertTrackFromTreeMuons(event, track, 1)
				tracks.append(values['sets'])
				infos.append(values['infos'])       

		outputFileName = fileName.split('/')[-1] + '.npz'

		np.savez_compressed(outputFileName,
							tracks=tracks,
							infos=infos)

		print 'Wrote', outputFileName

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

	def load_model(self, model_path):
		self.model = keras.models.load_model(model_path)

	def load_model_weights(self, weights_path):
		self.model.load_weights(weights_path)

	def evaluate_model(self, event, track):
		event = self.convertTrackFromTreeElectrons(event, track, 1) # class_label doesn't matter
		prediction = self.model.predict(np.reshape(event['sets'], (1, 100, 4)))
		#prediction = self.model.predict([np.reshape(event['sets'], (1, 100, 4)),np.reshape(event['infos'],(1,13))[:,[4,8,9]]])
		return prediction[:,1] # p(is electron)

	def evaluate_npy(self, fname, track_info=False, obj='sets'):

		data = np.load(fname, allow_pickle=True)

		if(data[obj].shape[0] == 0): return True, 0
		sets = data[obj][:,:self.max_hits]

		x = [sets]

		if(track_info):
			if obj == 'sets': info = data['info'][:,[4,8,9,12]]
			else: info = data[obj+'_info'][:,[4,8,9,12]]
			x.append(info)
			
		return False, self.model.predict(x,)

	def fit_generator(self, train_generator, val_generator=None, 
						epochs=10, monitor='val_loss',patience_count=10,
						metrics = ['accuracy'],
						outdir=""):

		self.model.compile(optimizer=optimizers.Adagrad(), loss='categorical_crossentropy', metrics=metrics)
		
		training_callbacks = [
			callbacks.EarlyStopping(monitor=monitor,patience=patience_count),
			callbacks.ModelCheckpoint(filepath=outdir+'model.{epoch}.h5',
											save_best_only=True,
											monitor=monitor,
											mode='auto')
		]

		if val_generator == None:
			self.training_history = self.model.fit(train_generator, 
											 validation_data=val_generator,
											 callbacks=training_callbacks,
											 epochs=epochs,
											 verbose=2)
		else:
			self.training_history = self.model.fit(train_generator, 
											 epochs=epochs,
											 verbose=2)

	def save_model(self, outputFileName):
		self.model.save(outputFileName)
		print 'Saved model in file:', outputFileName

	def save_weights(self, outputFileName):
		self.model.save_weights(outputFileName)
		print 'Saved model weights in file:', outputFileName

	def save_trainingHistory(self, outputFileName):
		with open(outputFileName, 'wb') as f:
			pickle.dump(self.training_history.history, f)
		print 'Saved training history in file:', outputFileName

	def displayTrainingHistory(self):
		acc = self.training_history.history['accuracy']
		val_acc = self.training_history.history['val_accuracy']

		loss = self.training_history.history['loss']
		val_loss = self.training_history.history['val_loss']

		epochs = range(1, len(acc) + 1)

		plt.plot(epochs, acc, 'bo', label='Training acc')
		plt.plot(epochs, val_acc, 'b', label='Validation acc')
		plt.title('Training and validation accuracy')
		plt.legend()

		plt.figure()
		plt.plot(epochs, loss, 'bo', label='Training loss')
		plt.plot(epochs, val_loss, 'b', label='Validation loss')
		plt.title('Training and validation loss')
		plt.legend()

		plt.show()

	def plot_trainingHistory(self,infile,outfile,metric='loss'):

		with open(infile,'rb') as f:
			history = pickle.load(f)

		loss = history[metric]
		val_loss = history['val_'+metric]

		epochs = range(1, len(loss) + 1)

		plt.figure()
		plt.plot(epochs, loss, 'bo', label='Training '+metric)
		plt.plot(epochs, val_loss, 'b', label='Validation '+metric)
		plt.title('Training and validation loss')
		plt.legend()

		plt.savefig(outfile)

	def save_metrics(self, infile, outfile, train_params):

		file = open(infile,'rb')
		history = pickle.load(file)
		if(len(history['val_loss']) == train_params['epochs']):
			val_loss = history['val_loss'][-1]
			val_acc = history['val_accuracy'][-1]
		else:
			i = len(history['val_loss']) - train_params['patience_count'] - 1
			val_loss = history['val_loss'][i]
			val_acc = history['val_accuracy'][i]
		file.close()
		metrics = {
			"val_loss":val_loss,
			"val_acc":val_acc
		}
		with open(outfile, 'wb') as f:
			pickle.dump(metrics, f)

	def save_kfoldMetrics(self, infiles, outfile, train_params):

		k = len(infiles)
		val_loss, val_acc = 0,0
		for infile in infiles:
			file = open(infile,'rb')
			history = pickle.load(file)
			if(len(history['val_loss']) == train_params['epochs']):
				val_loss += history['val_loss'][-1]
				val_acc += history['val_accuracy'][-1]
			else:
				i = len(history['val_loss']) - train_params['patience_count'] - 1
				val_loss += history['val_loss'][i]
				val_acc += history['val_accuracy'][i]
			file.close()

		val_loss /= k
		val_acc /= k
		metrics = {
			"val_loss":val_loss,
			"val_accuracy":val_acc
		}

		with open(outfile, 'wb') as f:
			pickle.dump(metrics, f)

	# expects [c1,c1] = TP, [c2,c2] = TN
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

	def calc_cm(self, true,predictions,dim=2):
		confusion_matrix = np.zeros((dim, dim))
		for t,p in zip(true, predictions):
			confusion_matrix[t,p] += 1
		return confusion_matrix

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

		