import os
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import optimizers, regularizers
import json
import random
import sys
import pickle
import datetime
import getopt
			
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, TimeDistributed, Masking, Input, Lambda, Activation, BatchNormalization

import utils
import validate
from generator import generator
from model import buildModel

# limit CPU usage
config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads = 2,   
								intra_op_parallelism_threads = 2,
								allow_soft_placement = True,
								device_count={'CPU': 2})
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

try:
	opts, args = getopt.getopt(sys.argv[1:], 
							"d:p:i:", 
							["dir=","params=","index="])
except getopt.GetoptError:
	print(utils.bcolors.RED+"USAGE: flow.py -d/--dir= output_directory -p/--params= parameters.npy -i/--index= parameter_index"+utils.bcolors.ENDC)
	sys.exit(2)

workDir = 'deepSets'
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
		print(utils.bcolors.RED+"ERROR: Index outside range or no parameter list passed"+utils.bcolors.ENDC)
		print(utils.bcolors.RED+"USAGE: flow.py -d/--dir= output_directory -p/--params= parameters.npy -i/--index= parameter_index"+utils.bcolors.ENDC)
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
dataDir = "/store/user/llavezzo/disappearingTracks/converted_deepSets100_failAllRecos/"
#logDir = "/home/llavezzo/work/cms/logs/"+ workDir +"_"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

run_validate = True
nTotE = 25000
val_size = 0.2
undersample_bkg = 0.5
v = 2
batch_size = 64
epochs = 1
patience_count = 10
monitor = 'val_loss'
metrics = ['accuracy']
#################################################

if(len(params) > 0):
	undersample_bkg = float(params[0])
	epochs = int(params[1])
	dataDir = str(params[2])

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
train_generator = generator(train_e_file_batches, train_bkg_file_batches, train_e_event_batches, train_bkg_event_batches, 
					batch_size, dataDir, False, True)
val_generator = generator(val_e_file_batches, val_bkg_file_batches, val_e_event_batches, val_bkg_event_batches, 
					batch_size, dataDir, False, True)

model = buildModel()

model.compile(optimizer=optimizers.Adam(), 
			  loss='categorical_crossentropy', 
			  metrics=metrics)

callbacks = [
	keras.callbacks.EarlyStopping(patience=patience_count),
	keras.callbacks.ModelCheckpoint(filepath=weightsDir+'model.{epoch}.h5',
									save_best_only=True,
									monitor=monitor,
									mode='auto')
	# tf.keras.callbacks.TensorBoard(log_dir=logDir, 
	#                                histogram_freq=0,
	#                                write_graph=False,
	#                                write_images=False)
]


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

utils.plot_history(history, plotDir, ['loss','accuracy'])
print(utils.bcolors.YELLOW+"Plotted history to "+plotDir+utils.bcolors.ENDC) 

if(run_validate): validate.run_validation(model, weightsDir+'lastEpoch.h5', outputDir, dataDir, plotDir, batch_size)