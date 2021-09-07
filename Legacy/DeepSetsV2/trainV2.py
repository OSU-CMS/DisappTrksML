import os, sys, getopt
import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import pickle
import datetime
			
import utils
import validate
from generator import generator2
from model import buildModel, buildModelWithEventInfo

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
	print(utils.bcolors.RED+"USAGE: train.py -d/--dir= output_directory -p/--params= parameters.npy -i/--index= parameter_index "+utils.bcolors.ENDC)
	sys.exit(2)

workDir = 'train'
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
	else: workDir = workDir[:-int(int(cnt*1.0/10)+1)] + str(cnt)

print(utils.bcolors.YELLOW+"Output directory: "+workDir+utils.bcolors.ENDC)
if(len(params) > 0): 
	print(utils.bcolors.YELLOW+"Using params"+utils.bcolors.ENDC, params, end=" ")
	print(utils.bcolors.YELLOW+"from file "+paramsFile+utils.bcolors.ENDC)

plotDir = workDir + '/plots/'
weightsDir = workDir + '/weights/'
outputDir = workDir + '/outputFiles/'

################config parameters################
dataDir = "C:/Users/llave/Documents/CMS/data/images_DYJetsToLL_v4_sets_electrons/"

run_validate = True
nTotE = 12000
val_size = 0.2
undersample_bkg = 0.9
v = 1
batch_size = 512
epochs = 10
patience_count = 10
monitor = 'val_loss'
metrics = ['accuracy']
#################################################

if(len(params) > 0):
	undersample_bkg = float(params[0])
	epochs = int(params[1])
	dataDir = str(params[2])

print("Running over", dataDir)

# create output directories
os.makedirs(workDir)
os.makedirs(plotDir)
os.makedirs(weightsDir)
os.makedirs(outputDir)

train_data, val_data = utils.prepare_data2(dataDir, nTotE, batch_size, val_size, undersample_bkg)  

# save the train and validation batches
np.savez_compressed(outputDir+"train_data", events = train_data[0], files = train_data[1], classes = train_data[2])
np.savez_compressed(outputDir+"val_data",  events = val_data[0], files = val_data[1], classes = val_data[2])

# initialize generators
train_generator = generator2(train_data, 
					batch_size, dataDir, True, True, False)
val_generator = generator2(val_data,
					batch_size, dataDir, True, True, True)

model = buildModelWithEventInfo(info_shape=5)
#model = buildModel()

model.compile(optimizer=keras.optimizers.Adam(), 
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

if(run_validate): validate.run_batch_validation(model, weightsDir+'lastEpoch.h5', outputDir, dataDir, plotDir)