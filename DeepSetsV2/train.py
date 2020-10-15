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
from generator import generator
from model import buildModel, buildModelWithEventInfo

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
dataDir = "/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v4_sets_muons/"
logDir = "/home/" + os.environ["USER"] + "/logs/"+ workDir +"_"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

run_validate = True
nTotE = 2500 
val_size = 0.2
undersample_bkg = 0.9
v = 1
batch_size = 128
epochs = 5
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

e_data, bkg_data = utils.prepare_data(dataDir, nTotE, batch_size, val_size, undersample_bkg)  

# save the train and validation batches
np.save(outputDir+"e_files_trainBatches", e_data[0])
np.save(outputDir+"e_events_trainBatches", e_data[1])
np.save(outputDir+"e_files_valBatches", e_data[2])
np.save(outputDir+"e_events_valBatches", e_data[3])
np.save(outputDir+"bkg_files_trainBatches", bkg_data[0])
np.save(outputDir+"bkg_events_trainBatches", bkg_data[1])
np.save(outputDir+"bkg_files_valBatches", bkg_data[2])
np.save(outputDir+"bkg_events_valBatches", bkg_data[3])

# initialize generators
train_generator = generator(e_data[0], e_data[1], bkg_data[0], bkg_data[1], 
					batch_size, dataDir, False, True, True)
val_generator = generator(e_data[2], e_data[3], bkg_data[2], bkg_data[3], 
					batch_size, dataDir, False, True, True)

model = buildModelWithEventInfo(info_shape=5)

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