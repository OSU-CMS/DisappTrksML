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
from collections import Counter
			
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, TimeDistributed, Masking, Input, Lambda, Activation, BatchNormalization

import utils
import validate
from generatorV2 import generator
from model import buildModel

# limit CPU usage
# config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads = 4,   
# 								intra_op_parallelism_threads = 4,
# 								allow_soft_placement = True,
# 								device_count={'CPU': 4})
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

try:
	opts, args = getopt.getopt(sys.argv[1:], 
							"d:p:i:", 
							["dir=","params=","index="])
except getopt.GetoptError:
	print(utils.bcolors.RED+"USAGE: flow.py -d/--dir= output_directory -p/--params= parameters.npy -i/--index= parameter_index "+utils.bcolors.ENDC)
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
logDir = "/home/" + os.environ["USER"] + "/logs/"+ workDir +"_"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

run_validate = True
nTotE = 25000 
val_size = 0.2
undersample_bkg = -1
v = 1
batch_size = 512
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
os.makedirs(workDir)
os.makedirs(plotDir)
os.makedirs(weightsDir)
os.makedirs(outputDir)
os.makedirs(logDir)

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
nBatches = int(np.floor((nTotE + nTotBkg)*1.0/batch_size))

assert nTotE + nTotBkg >= nBatches*batch_size, "Not enough events to fill batches"

# fill lists of all events and files
bkg_batches = []
for file, nEvents in bkgCounts.items():
	for evt in range(nEvents):
		bkg_batches.append([0,file,evt])
e_batches = []
for file, nEvents in eCounts.items():
	for evt in range(nEvents):
		e_batches.append([1,file,evt])
p = np.random.permutation(len(e_batches))[:nTotE]
e_batches = np.array(e_batches)[p]
p = np.random.permutation(len(bkg_batches))[:nTotBkg]
bkg_batches = np.array(bkg_batches)[p]

# (batch,event,(class,file,event index))
events = np.concatenate((e_batches,bkg_batches))[:int(batch_size*nBatches)]
batches = np.split(events,nBatches)

# train/validation split
train_batches, val_batches = train_test_split(batches, test_size=val_size, random_state=42)

labels = np.array(train_batches)[:,:,0].flatten()
eTrain = len(labels[np.where(labels==1)])
bkgTrain = len(labels[np.where(labels==0)])
labels = np.array(val_batches)[:,:,0].flatten()
eVal = len(labels[np.where(labels==1)])
bkgVal = len(labels[np.where(labels==0)])

print("\t\tElectrons\tBackground\te/(e+bkg)")
print("Requested:\t"+str(nTotE)+"\t\t"+str(nTotBkg)+"\t\t"+str(round(nTotE*1.0/(nTotE+nTotBkg),5)))
print("Training on:\t"+str(eTrain)+"\t\t"+str(bkgTrain)+"\t\t"+str(round(eTrain*1.0/(eTrain+bkgTrain),5)))
print("Validating on:\t"+str(eVal)+"\t\t"+str(bkgVal)+"\t\t"+str(round(eVal*1.0/(eVal+bkgVal),5)))
print("Dataset:\t"+str(availableE)+"\t\t"+str(availableBkg)+"\t\t"+str(round(fE,5)))

# save the train and validation batches
np.save(outputDir+"batches", batches)

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
train_generator = generator(train_batches, batch_size, dataDir, False, True)
val_generator = generator(val_batches, batch_size, dataDir, False, True)


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