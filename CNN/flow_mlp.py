import os
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from tensorflow.keras.applications import VGG19
import json
import random
import sys
import pickle
import datetime
import getopt

import utils
import validate_mlp

def build_cnn(input_shape=(40,40,3), batch_norm = False, filters=[128,256], 
				output_bias=0, metrics=['accuracy']):
	
	model = keras.Sequential()
	
	model.add(keras.layers.Conv2D(filters[0], kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	if(batch_norm): model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Dropout(0.2))   

	for layer in range(1,len(filters)):
		model.add(keras.layers.Conv2D(filters[layer], (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)))
		model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
		if(batch_norm): model.add(keras.layers.BatchNormalization())
		model.add(keras.layers.Dropout(0.2))

	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)))
	model.add(keras.layers.Dropout(0.4))

	model.add(keras.layers.Dense(1, activation='sigmoid',bias_initializer=keras.initializers.Constant(output_bias)))

	print(model.summary())

	return model

def build_mlp(input_dim = 1):
	model = keras.Sequential()
	model.add(keras.layers.Dense(8, input_dim=input_dim, activation='relu'))
	model.add(keras.layers.Dense(4, activation='relu'))
	return model

def build_VGG19(input_shape):
	
	base_model = VGG19(input_shape = input_shape, 
					include_top=False,
					weights=None,
					classifier_activation="sigmoid")

	x = base_model.output
	x = keras.layers.GlobalAveragePooling2D()(x)
	x = keras.layers.Dense(1024, activation='relu')(x)
	predictions = keras.layers.Dense(1, activation='sigmoid')(x)
	model = keras.models.Model(inputs=base_model.input, outputs=predictions)

	# first: train only the top layers (which were randomly initialized)
	# i.e. freeze all convolutional InceptionV3 layers
	# for layer in base_model.layers:
	#     layer.trainable = False

	# compile the model (should be done *after* setting layers to non-trainable)
	model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
				optimizer='adam',
				metrics=metrics)

	return model

# generate batches of images from files
class generator(keras.utils.Sequence):
  
	def __init__(self, batchesE, batchesBkg, indicesE, indicesBkg, 
				batch_size, dataDir, shuffle=True):
		self.batchesE = batchesE
		self.batchesBkg = batchesBkg
		self.indicesE = indicesE
		self.indicesBkg = indicesBkg
		self.batch_size = batch_size
		self.dataDir = dataDir
		self.shuffle = shuffle

	def __len__(self):
		return len(self.batchesE)

	def __getitem__(self, idx) :

		filenamesE = self.batchesE[idx]
		filenamesBkg = self.batchesBkg[idx]
		indexE = self.indicesE[idx]
		indexBkg = self.indicesBkg[idx]

		lastFile = len(filenamesE)-1
		filenamesE.sort()
		for iFile, file in enumerate(filenamesE):
			
			if(file == -1): 
				e_images = np.array([])
				continue
			print("Loading File: " + str(file))
			e_file = np.load(self.dataDir+'e_0p25_'+str(file)+'.npz')
			if(iFile == 0 and iFile != lastFile):
				e_images = e_file['images'][indexE[0]:]
				e_infos = e_file['infos'][indexE[0]:, 7]

			elif(iFile == lastFile and iFile != 0):
				e_images = np.vstack((e_images,e_file['images'][:indexE[1]+1]))
				e_infos = np.concatenate((e_infos,e_file['infos'][:indexE[1]+1, 7]))

			elif(iFile == 0 and iFile == lastFile):
				e_images = e_file['images'][indexE[0]:indexE[1]+1]
				e_infos = e_file['infos'][indexE[0]:indexE[1]+1, 7]

			elif(iFile != 0 and iFile != lastFile):
				e_images = np.vstack((e_images,e_file['images']))
				e_infos = np.concatenate((e_infos, e_file['infos'][:,7]))
		
		lastFile = len(filenamesBkg)-1
		filenamesBkg.sort()
		for iFile, file in enumerate(filenamesBkg):

			bkg_file = np.load(self.dataDir+'bkg_0p25_'+str(file)+'.npz')
			print("Loading File: " + str(file))

			if(iFile == 0 and iFile != lastFile):
				bkg_images = bkg_file['images'][indexBkg[0]:,:]
				bkg_infos = bkg_file['infos'][indexBkg[0]:,7]

			elif(iFile == lastFile and iFile != 0):
				bkg_images = np.vstack((bkg_images,bkg_file['images'][:indexBkg[1]+1]))
				bkg_infos = np.concatenate((bkg_infos,bkg_file['infos'][:indexBkg[1]+1, 7]))

			elif(iFile == 0 and iFile == lastFile):
				bkg_images = bkg_file['images'][indexBkg[0]:indexBkg[1]+1]
				bkg_infos = bkg_file['infos'][indexBkg[0]:indexBkg[1]+1, 7]

			elif(iFile != 0 and iFile != lastFile):
				bkg_images = np.vstack((bkg_images,bkg_file['images']))
				bkg_infos = np.concatenate((bkg_infos, bkg_file['infos'][:, 7]))
		
		numE = e_images.shape[0]
		numBkg = self.batch_size-numE
		bkg_images = bkg_images[:numBkg]
		#bkg_eta = bkg_infos[:, 7]
		bkg_eta = bkg_infos[:numBkg]

		# shuffle and select appropriate amount of electrons, bkg
		indices = list(range(bkg_images.shape[0]))
		random.shuffle(indices)
		bkg_images = bkg_images[indices,2:]
		bkg_eta = bkg_eta[indices]

		if(numE != 0):
			indices = list(range(e_images.shape[0]))
			random.shuffle(indices)
			e_images = e_images[indices,2:]
			#e_eta = e_infos[indices, 7]
			e_eta = e_infos[indices]



		# concatenate images and suffle them, create labels
		if(numE != 0): 
			batch_x = np.vstack((e_images,bkg_images))
			etas = np.concatenate((e_eta, bkg_eta))
		else: 
			batch_x = bkg_images
			etas = np.array(bkg_eta)
		
		batch_y = np.concatenate((np.ones(numE),np.zeros(numBkg)))

		indices = list(range(batch_x.shape[0]))
		random.shuffle(indices)

		batch_x = batch_x[indices[:self.batch_size],:]
		batch_x = np.reshape(batch_x,(self.batch_size,40,40,4))
		batch_x = batch_x[:,:,:,[0,2,3]]
	
		etas = etas[indices[:self.batch_size]]	
		
		max_eta = np.max(np.abs(etas))
		etas = np.abs(etas / max_eta)
		batch_y = batch_y[indices[:self.batch_size]]

		return [batch_x, etas], batch_y
	
	def on_epoch_end(self):
		if(self.shuffle):
			indexes = np.arange(len(self.batchesE))
			np.random.shuffle(indexes)
			self.batchesE = batchesE[indexes]
			self.batchesBkg = batchesBkg[indexes]
			self.indicesE = indicesE[indexes]
			self.indicesBkg = indicesBkg[indexes]
			

if __name__ == "__main__":

	# limit CPU usage
	config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads = 4,   
									intra_op_parallelism_threads = 4,
									allow_soft_placement = True,
									device_count={'CPU': 4})
	tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

	# suppress warnings
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

	try:
		opts, args = getopt.getopt(sys.argv[1:], 
								"d:p:i:", 
								["dir=","params=","index="])
	except getopt.GetoptError:
		print(utils.bcolors.RED+"USAGE: flow_mlp.py -d/--dir= output_directory -p/--params= parameters.npy -i/--index= parameter_index"+utils.bcolors.ENDC)
		sys.exit(2)

	workDir = 'mlp_results_9_1_small'
	paramsFile = ""
	params = []
	paramsIndex = 0
	for opt, arg in opts:
		if(opt in ('-d','--dir')):
			workDir = str(arg)
		elif(opt in ('-p','--params')):
			paramsFile = str(arg)
		elif(opt in ('-i','--index')):
			paramsIndex = int(arg)

	if(len(paramsFile)>0):
		try:
			params = np.load(str(paramsFile), allow_pickle=True)[paramsIndex]
		except:
			print(str(paramsFile))
			print(utils.bcolors.RED+"ERROR: Index outside range or no parameter list passed"+utils.bcolors.ENDC)
			print(utils.bcolors.RED+"USAGE: flow_mlp.py -d/--dir= output_directory -p/--params= parameters.npy -i/--index= parameter_index"+utils.bcolors.ENDC)
			sys.exit(2)
		workDir = workDir + "_p" + str(paramsIndex)
	cnt=0
	while(os.path.isdir(workDir)):
		cnt+=1
		if(cnt==1): workDir = workDir+"_"+str(cnt)
		else: workDir = workDir[:-1] + str(cnt)
	print(utils.bcolors.YELLOW+"Output directory: "+workDir+utils.bcolors.ENDC)
	if(len(params) > 0): 
		print(utils.bcolors.YELLOW+"Using params"+utils.bcolors.ENDC, params, end=" ")
		print(utils.bcolors.YELLOW+"from file "+paramsFile+utils.bcolors.ENDC)
	
	plotDir = workDir + '/plots/'
	weightsDir = workDir + '/weights/'
	outputDir = workDir + '/outputFiles/'

	################config parameters################
	"""
	nTotE: number of electron events to use
	oversample_e: (NOT WORKING) fraction of electron events per train batch, set to -1 if it's not needed
	undersample_bkg: fraction of backgruond events per train batch, set to -1 if it's not needed
	v: verbosity
	patience_count: after how many epochs to stop if monitored variable doesn't improve
	monitor: which variable to monitor with patience_count
	"""

	dataDir = "/store/user/mcarrigan/disappearingTracks/electron_selection_tanh_5gt0p5/"
	#logDir = "/home/llavezzo/work/cms/logs/"+ workDir +"_"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

	run_validate = True
	nTotE = 1000
	val_size = 0.2
	undersample_bkg = 0.5
	oversample_e = -1   
	filters = [128, 256]
	batch_norm = True
	v = 2
	batch_size = 256
	epochs = 2
	patience_count = 20
	monitor = 'val_loss'
	class_weights = False  
	metrics = [keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
	#################################################

	if(len(params) > 0):
		filters = params[0]
		class_weights = bool(params[1])
		undersample_bkg = float(params[2])
		epochs = int(params[3])

	# create output directories
	os.system('mkdir '+str(workDir))
	os.system('mkdir '+str(plotDir))
	os.system('mkdir '+str(weightsDir))
	os.system('mkdir '+str(outputDir))

	# import count dicts
	with open(dataDir+'eCounts.pkl', 'rb') as f:
		eCounts = pickle.load(f)
	with open(dataDir+'bkgCounts.pkl', 'rb') as f:
		bkgCounts = pickle.load(f)

	# count how many events are in the files for each class
	availableE = sum(list(eCounts.values()))
	availableBkg = sum(list(bkgCounts.values()))

	# fractions for each class for the total dataset
	fE = availableE*1.0/(availableE + availableBkg)
	fBkg = availableBkg*1.0/(availableE + availableBkg)

	# calculate how many total background events for the requested electrons
	# to keep the same fraction of events, or under sample
	nTotBkg = int(nTotE*1.0*availableBkg/availableE)
	if(undersample_bkg!=-1): nTotBkg = int(nTotE*1.0*undersample_bkg/(1-undersample_bkg))

	# can't request more events than we have
	if(nTotE > availableE): sys.exit("ERROR: Requested more electron events than are available")
	if(nTotBkg > availableBkg): sys.exit("ERROR: Requested more electron events than available")

	# batches per epoch
	nBatches = int((nTotE + nTotBkg)*1.0/batch_size)

	# count how many e/bkg events in each batch
	ePerBatch = np.zeros(nBatches)
	iBatch = 0
	while np.sum(ePerBatch) < nTotE:
		ePerBatch[iBatch]+=1
		iBatch+=1
		if(iBatch == nBatches): iBatch = 0
	bkgPerBatch = np.asarray([batch_size-np.min(ePerBatch)]*nBatches)
	ePerBatch = ePerBatch.astype(int)
	bkgPerBatch = bkgPerBatch.astype(int)

	# fill lists of all events and files
	b_events, b_files = [], []
	for file, nEvents in bkgCounts.items():
		for evt in range(nEvents):
			b_events.append(evt)
			b_files.append(file)
	e_events, e_files = [], []
	for file, nEvents in eCounts.items():
		for evt in range(nEvents):
			e_events.append(evt)
			e_files.append(file)

	# make batches
	bkg_event_batches, bkg_file_batches = utils.make_batches(b_events, b_files, bkgPerBatch, nBatches)
	e_event_batches, e_file_batches = utils.make_batches(e_events, e_files, ePerBatch, nBatches)

	# train/validation split
	train_e_event_batches, val_e_event_batches, train_e_file_batches, val_e_file_batches = train_test_split(e_event_batches, e_file_batches, test_size=val_size, random_state=42)
	train_bkg_event_batches, val_bkg_event_batches, train_bkg_file_batches, val_bkg_file_batches = train_test_split(bkg_event_batches, bkg_file_batches, test_size=val_size, random_state=42)

	# count events in each batch
	nSavedETrain = utils.count_events(train_e_file_batches, train_e_event_batches, eCounts)
	nSavedEVal = utils.count_events(val_e_file_batches, val_e_event_batches, eCounts)
	nSavedBkgTrain = utils.count_events(train_bkg_file_batches, train_bkg_event_batches, bkgCounts)
	nSavedBkgVal = utils.count_events(val_bkg_file_batches, val_bkg_event_batches, bkgCounts)

	# add background events to validation data
	# to keep ratio e/bkg equal to that in original dataset
	if(nSavedEVal*1.0/(nSavedEVal+nSavedBkgVal) > fE):
		nBkgToLoad = int(nSavedEVal*(1-fE)/fE-nSavedBkgVal)
		lastFile = bkg_file_batches[-1][-1]

		b_events, b_files = [], []
		reached = False
		for file, nEvents in bkgCounts.items():
			if(int(file) != lastFile and not reached): continue
			else: reached = True

			for evt in range(nEvents):
				b_events.append(evt)
				b_files.append(file)

		# make batches of same size with bkg files
		nBatchesAdded = int(nBkgToLoad*1.0/batch_size)
		bkgPerBatch = [batch_size]*nBatchesAdded
			   
		bkg_event_batches_added, bkg_file_batches_added = utils.make_batches(b_events, b_files, bkgPerBatch, nBatchesAdded)

		nAddedBkg = utils.count_events(bkg_file_batches, bkg_event_batches, bkgCounts)

		# add the bkg and e events to rebalance val data
		filler_events = [[0,0]]*nBatchesAdded
		filler_files = [list(set([-1])) for _ in range(nBatchesAdded)]
		val_bkg_event_batches = np.concatenate((val_bkg_event_batches,bkg_event_batches_added))
		val_bkg_file_batches = val_bkg_file_batches + bkg_file_batches_added
		val_e_event_batches = np.concatenate((val_e_event_batches,filler_events))
		val_e_file_batches = val_e_file_batches + filler_files

		# re count
		nSavedEVal = utils.count_events(val_e_file_batches, val_e_event_batches, eCounts)
		nSavedBkgVal = utils.count_events(val_bkg_file_batches, val_bkg_event_batches, bkgCounts)


	print("\t\tElectrons\tBackground\te/(e+bkg)")
	print("Requested:\t"+str(nTotE)+"\t\t"+str(nTotBkg)+"\t\t"+str(round(nTotE*1.0/(nTotE+nTotBkg),5)))
	print("Training on:\t"+str(nSavedETrain)+"\t\t"+str(nSavedBkgTrain)+"\t\t"+str(round(nSavedETrain*1.0/(nSavedETrain+nSavedBkgTrain),5)))
	print("Validating on:\t"+str(nSavedEVal)+"\t\t"+str(nSavedBkgVal)+"\t\t"+str(round(nSavedEVal*1.0/(nSavedEVal+nSavedBkgVal),5)))
	print("Dataset:\t"+str(availableE)+"\t\t"+str(availableBkg)+"\t\t"+str(round(fE,5)))
	
	# save the train and validation batches
	np.save(outputDir+"e_files_trainBatches", train_e_file_batches)
	np.save(outputDir+"e_events_trainBatches", train_e_event_batches)
	np.save(outputDir+"e_files_valBatches", val_e_file_batches)
	np.save(outputDir+"e_events_valBatches", val_e_event_batches)
	np.save(outputDir+"bkg_files_trainBatches", train_bkg_file_batches)
	np.save(outputDir+"bkg_events_trainBatches", train_bkg_event_batches)
	np.save(outputDir+"bkg_files_valBatches", val_bkg_file_batches)
	np.save(outputDir+"bkg_events_valBatches", val_bkg_event_batches)

	# FIXME: not implemented yet
	# oversample the training electron files if oversample_e != -1
	# nElectronsOversampled = int(np.ceil(nSavedETrain*oversample_e)) - nSavedETrain
	# ovsFiles = list([file for batch in train_e_file_batches for file in batch])
	# random.shuffle(ovsFiles)
	# for i,batch in enumerate(train_e_file_batches):
	#     nElectronsThisBatch = 0
	#     for file in batch: nElectronsThisBatch+=eCounts[file]
	#     while nElectronsThisBatch < nElectronsPerBatchOversampled:
	#         randFile = ovsFiles[random.randint(0,len(ovsFiles)-1)]
	#         trainBatchesE[i].append(randFile)
	#         nElectronsThisBatch += eCounts[randFile]
	# if(oversample_e != -1):
	#     print("Oversampling:")
	#     print("\t Number of electrons per batch:",nElectronsPerBatchOversampled)
	#     print("\t",len(trainBatchesE),"batches of files (approx.",nElectronsPerBatchOversampled*len(trainBatchesE),"electron and",(batch_size-nElectronsPerBatchOversampled)*len(trainBatchesE), "background events)")

	# initialize generators
	train_generator = generator(train_e_file_batches, train_bkg_file_batches, train_e_event_batches, train_bkg_event_batches, batch_size, dataDir, True)
	val_generator = generator(val_e_file_batches, val_bkg_file_batches, val_e_event_batches, val_bkg_event_batches, batch_size, dataDir, True)
	

	# initialize output bias
	output_bias = np.log(nSavedETrain/nSavedBkgTrain)

	cnn = build_cnn(input_shape = (40,40,3), filters = [128, 256], batch_norm=batch_norm, output_bias=output_bias, metrics=metrics)
	mlp = build_mlp(input_dim=1)

	combinedInput = keras.layers.concatenate([cnn.output, mlp.output])
	x = keras.layers.Dense(4, activation="relu")(combinedInput)
	x = keras.layers.Dense(1, activation="sigmoid")(x)
	model = keras.models.Model(inputs=[cnn.input, mlp.input], outputs=x)
	model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer="adam", metrics=metrics)

	callbacks = [
		#keras.callbacks.EarlyStopping(patience=patience_count),
		keras.callbacks.ModelCheckpoint(filepath=weightsDir+'model.{epoch}.h5',
										#save_best_only=True,
										monitor=monitor,
										mode='auto')
	   # tf.keras.callbacks.TensorBoard(log_dir=logDir, 
	   #                                 histogram_freq=0,
	   #                                 write_graph=False,
	   #                                 write_images=False)
	]

	if(class_weights):

		# class weights
		weight_for_0 = (1/nSavedBkgTrain)*(nSavedBkgTrain+nSavedETrain)/2.0
		weight_for_1 = (1/nSavedETrain)*(nSavedBkgTrain+nSavedETrain)/2.0
		class_weight = {0: weight_for_0, 1: weight_for_1}

		history = model.fit(train_generator, 
					epochs = epochs,
					verbose= v,
					validation_data=val_generator,
					callbacks=callbacks,
					class_weight=class_weight)

	else:
		history = model.fit(train_generator,
					epochs = epochs,
					verbose= v,
					validation_data=val_generator,
					callbacks=callbacks)
		   
	model.save_weights(weightsDir+'lastEpoch.h5')
	print(utils.bcolors.GREEN+"Saved weights to "+weightsDir+utils.bcolors.ENDC)

	# save and plot history file
	with open(outputDir+'history.pkl', 'wb') as f:
		pickle.dump(history.history, f)
	print(utils.bcolors.GREEN+"Saved history, train and validation files to "+outputDir+utils.bcolors.ENDC)

	utils.plot_history(history, plotDir, ['loss','recall','precision','auc'])
	print(utils.bcolors.YELLOW+"Plotted history to "+plotDir+utils.bcolors.ENDC) 

	if(run_validate): validate_mlp.validate_mlp(model, weightsDir+'lastEpoch.h5', outputDir, dataDir, plotDir, batch_size)
