#!/usr/bin/env python
from DisappTrksML.SiameseNetwork.architecture import *

class SiameseNetwork(GeneralArchitecture):
	
	def __init__(self, eta_range=0.25, phi_range=0.25, max_hits=100,
				 phi_layers=[64, 64, 256], f_layers=[64, 64, 64],
				 track_info_indices=None,
				 ):
		GeneralArchitecture.__init__(self, eta_range, phi_range, max_hits, phi_layers, f_layers, track_info_indices)
		self.input_shape = (self.max_hits, 7)
		if self.track_info_indices is None: self.track_info_shape = 0
		else: self.track_info_shape = len(track_info_indices)

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

		class0, class1, class2, class3 = [], [], [], []

		for event in inputTree:
			eventPasses, trackPasses = self.eventSelectionTraining(event)
			if not eventPasses: continue

			for i, track in enumerate(event.tracks):
				if not trackPasses[i]: continue

				if isGenMatched(event, track, 13):

					if abs(track.deltaRToClosestMuon) < 0.15:
						values = self.convertTrackFromTree(event, track, 0)
						class0.append(values['sets'])

					else:
						values = self.convertTrackFromTree(event, track, 1)
						class1.append(values['sets'])					

				else:

					if abs(track.deltaRToClosestMuon) < 0.15:
						values = self.convertTrackFromTree(event, track, 2)
						class2.append(values['sets'])

					else:
						values = self.convertTrackFromTree(event, track, 3)
						class3.append(values['sets'])

		outputFileName = fileName.split('/')[-1] + '.npz'

		if len(class0) != 0 or len(class1) != 0 or len(class2) != 0 or len(class3) != 0:

			np.savez_compressed(outputFileName,
								class0=class0,
								class1=class1,
								class2=class2,
								class3=class3)

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

				values = self.convertTrackFromTree(event, track, 1)
				tracks.append(values['sets'])
				infos.append(values['infos'])

		outputFileName = fileName.split('/')[-1] + '.npz'

		if len(tracks) > 0:
			np.savez_compressed(outputFileName,
								sets=tracks,
								infos=infos)

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
								signal=signal)

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
		deepset_inputs_r = Input(shape=self.input_shape)
		latent_space_r = phi_model(deepset_inputs_r)
		if(self.track_info_shape == 0): 
			deepset_outputs_r = f_model(latent_space_r)
		else: 
			info_inputs_r = Input(shape=(self.track_info_shape,))
			deepset_inputs_withInfo_r = concatenate([latent_space_r,info_inputs_r])
			deepset_outputs_r = f_model(deepset_inputs_withInfo_r)

		# build the DeepSets architecture
		deepset_inputs_l = Input(shape=self.input_shape)
		latent_space_l = phi_model(deepset_inputs_l)
		if(self.track_info_shape == 0): 
			deepset_outputs_l = f_model(latent_space_l)
		else: 
			info_inputs_l = Input(shape=(self.track_info_shape,))
			deepset_inputs_withInfo_l = concatenate([latent_space_l,info_inputs_l])
			deepset_outputs_l = f_model(deepset_inputs_withInfo_l)

		# Add a customized layer to compute the absolute difference between the encodings
		L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
		L1_distance = L1_layer([deepset_outputs_l, deepset_outputs_r])

		# Add a dense layer with a sigmoid unit to generate the similarity score
		prediction = Dense(1,activation='sigmoid',bias_initializer=initializers.RandomNormal(mean=0.5, stddev=0.01))(L1_distance)
	
		# Connect the inputs with the outputs
		if(self.track_info_shape == 0): model = Model(inputs=[deepset_inputs_l, deepset_inputs_r],outputs=prediction)
		else: model = Model(inputs=[[deepset_inputs_l, info_inputs_l],[deepset_inputs_r, info_inputs_r]],outputs=prediction)

		print(model.summary())

		self.model = model
		self.model.compile(optimizer=optimizers.Adam(lr = 0.00006), loss='binary_crossentropy')

	def load_data(self, fname, nEvents, nClasses):

		data = np.load(fname, allow_pickle=True)
		X = None
		for i in range(nClasses):
			iClass = data[i][:nEvents]
			iClass = np.reshape(iClass, (1, nEvents, 20, 7))
			if X is None: X = iClass
			else: X = np.vstack((X, iClass)) 

		print "Loaded {0} with shape {1}".format(fname, X.shape)
		return X

	def add_data(self, X_train=None, X_val=None, X_val_ref=None):

		# events to train on
		self.X_train = X_train

		# events to evaluate
		self.X_val = X_val 		

		# reference set, events to compare X_val with
		self.X_val_ref = X_val_ref		

	def get_batch(self, batch_size, use_test_data=False):
	
		# For each batch shuffle
		if use_test_data:
			n_classes, n_examples, n_hits, n_features = self.X_val.shape
		else:
			n_classes, n_examples, n_hits, n_features = self.X_train.shape
		classes = list(range(n_classes))
		examples = list(range(n_examples))
	
		# Each trial shuffle our classes and classes 
		random.shuffle(classes)
		random.shuffle(examples)
		targets = np.zeros((batch_size,))
		test_images = np.zeros((batch_size,n_hits, n_features))
		support_images = np.zeros((batch_size,n_hits, n_features))
	
		# Make sure the batch size is < half the classes
		# if batch_size < n_classes//2:
		if True:
			half_batch = batch_size//2
		
			# Get the indices for the 1st half - which are pairs from same class
			test_class_indices = classes[0:half_batch]
			test_example_index = examples[0]
			support_class_indices = classes[0:half_batch]
			support_example_index = examples[1]
			
	
			# Are we generating these batches using validation or training data?
			if use_test_data:
				test_images[0:half_batch,:,:] = self.X_val[test_class_indices,test_example_index,:,:]
				support_images[0:half_batch,:,:] = self.X_val[support_class_indices,support_example_index,:,:]
			else:
				test_images[0:half_batch,:,:] = self.X_train[test_class_indices,test_example_index,:,:]
				support_images[0:half_batch,:,:] = self.X_train[support_class_indices,support_example_index,:,:]
			targets[0:half_batch] = 1
			
			# Get the indices for the 2nd half - which are pairs from different classes
			test_class_indices = classes[half_batch:batch_size]
			test_example_index = examples[0]
			support_class_indices = classes[batch_size:batch_size+half_batch]
			support_example_index = examples[1]
			
			if use_test_data:
				test_images[half_batch:batch_size,:,:] = self.X_val[test_class_indices,test_example_index,:,:]
				support_images[half_batch:batch_size,:,:] = self.X_val[support_class_indices,support_example_index,:,:]
			else:
				test_images[half_batch:batch_size,:,:] = self.X_train[test_class_indices,test_example_index,:,:]
				support_images[half_batch:batch_size,:,:] = self.X_train[support_class_indices,support_example_index,:,:]
			targets[half_batch:batch_size] = 0
			
			# Reshape
			test_images = test_images.reshape(batch_size, n_hits, n_features)
			support_images = support_images.reshape(batch_size, n_hits, n_features)

		
		# Now shuffle coherently
		targets, test_images, support_images = shuffle(targets, test_images, support_images)
		pairs = [test_images, support_images]

		return pairs, targets

	def make_oneshot(self, N, use_test_data=True):
	
		# For each batch shuffle
		if use_test_data:
			n_classes, n_examples, n_hits, n_features = self.X_val.shape
		else:
			n_classes, n_examples, n_hits, n_features = self.X_train.shape
		classes = list(range(n_classes))
		examples = list(range(n_examples))
	
		# Each trial shuffle our classes and classes 
		random.shuffle(classes)
		random.shuffle(examples)
	
		# Get support indices
		support_class_indices = classes[0:N]
		support_example_indices = examples[0:N]
	
		# Get the images to test
		test_class_index = classes[0]
		test_example_index = examples[0]
	
		# Now get another example (but different) of the test
		test_class_index_other = classes[0]
		test_example_index_other = examples[1]
	
		# The first in our support sample is the correct class
		support_class_indices[0] = test_class_index_other
		support_example_indices[0] = test_example_index_other
		targets = np.zeros((N,))
		targets[0] = 1
	
		# Now form our images
		if use_test_data:
			test_images = np.asarray([self.X_val[test_class_index,test_example_index,:,:]]*N)
			test_images = test_images.reshape(N, n_hits, n_features)
			support_images = self.X_val[support_class_indices,support_example_indices,:,:]
			support_images = support_images.reshape(N, n_hits, n_features)
		else:
			test_images = np.asarray([self.X_train[test_class_index,test_example_index,:,:]]*N)
			test_images = test_images.reshape(N, n_hits, n_features)
			support_images = self.X_train[support_class_indices,support_example_indices,:,:]
			support_images = support_images.reshape(N, n_hits, n_features)
			
		# Form return
		pairs = [test_images, support_images]

		return pairs, targets, test_class_index

	def make_oneshot_val(self, N, use_test_data=True):
	
		# For each batch shuffle
		if use_test_data:
			n_classes, n_examples, n_hits, n_features = self.X_val.shape
			assert self.X_val.shape == self.X_val_ref.shape, "X_val and X_val_ref must have same shape"
		else:
			n_classes, n_examples, n_hits, n_features = self.X_train.shape
		classes = list(range(n_classes))
		examples = list(range(n_examples))
	
		# Each trial shuffle our classes and classes 
		random.shuffle(classes)
		random.shuffle(examples)
	
		# Get support indices
		support_class_indices = classes[0:N]
		support_example_indices = examples[0:N]
	
		# Get the images to test
		test_class_index = classes[0]
		test_example_index = examples[0]
	
		# Now get another example (but different) of the test
		test_class_index_other = classes[0]
		test_example_index_other = examples[1]
	
		# The first in our support sample is the correct class
		support_class_indices[0] = test_class_index_other
		support_example_indices[0] = test_example_index_other
		targets = np.zeros((N,))
		targets[0] = 1
	
		# Now form our images
		if use_test_data:
			test_images = np.asarray([self.X_val[test_class_index,test_example_index,:,:]]*N)
			test_images = test_images.reshape(N, n_hits, n_features)
			support_images = self.X_val_ref[support_class_indices,support_example_indices,:,:]
			support_images = support_images.reshape(N, n_hits, n_features)
		else:
			test_images = np.asarray([self.X_train[test_class_index,test_example_index,:,:]]*N)
			test_images = test_images.reshape(N, n_hits, n_features)
			support_images = self.X_train[support_class_indices,support_example_indices,:,:]
			support_images = support_images.reshape(N, n_hits, n_features)
			
		# Form return
		pairs = [test_images, support_images]

		return pairs, targets, test_class_index