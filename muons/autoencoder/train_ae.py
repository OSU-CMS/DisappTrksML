import warnings
warnings.filterwarnings('ignore')
import glob, os, sys
from datetime import datetime
import numpy as np
import pickle

import tensorflow as tf

from AE import AE
from generator import DataGenerator

if False:
	# limit CPU usage
	config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads = 4,   
									intra_op_parallelism_threads = 4,
									allow_soft_placement = True,
									device_count={'CPU': 4})
	tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

#######

backup_suffix = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
outdir = "train_"+backup_suffix+"/"
if(len(sys.argv)>1):
	input_params = np.load("params.npy",allow_pickle=True)[int(sys.argv[1])]
	outdir = input_params[0]

model_params = {
	'maxHits':20
}
generator_params = {
	'input_dir' : '/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_muons/',
	'batch_size' : 256,
	'shuffle' : True,
	'maxHits' : 20,
	'flatten' : True
}
train_params = {
	'epochs':10,
	'outdir':outdir,
	'patience_count':5,
	'monitor':'val_loss'
}

if(not os.path.isdir(outdir)): os.mkdir(outdir)

arch = AE(**model_params)
arch.buildModel()

inputFiles = glob.glob(generator_params['input_dir']+'images_*.root.npz')
inputIndices = np.array([f.split('images_')[-1][:-9] for f in inputFiles])
nFiles = len(inputIndices)
print('Found', nFiles, 'input files')

file_ids = {
	'train'      : inputIndices[:100],
	'validation' : inputIndices[500:510]
}

train_generator = DataGenerator(file_ids['train'], **generator_params)
val_generator = DataGenerator(file_ids['validation'], **generator_params)

arch.fit_generator(train_generator=train_generator, 
				   val_generator=val_generator, 	
					**train_params)

arch.save_trainingHistory(train_params['outdir']+'trainingHistory.pkl')
arch.plotHistory('ae_train/trainingHistory.pkl','ae_train/trainingHistory.png','loss')
arch.save_weights(train_params['outdir']+'model_weights.h5')
arch.save_model(train_params['outdir']+'model.h5')

# events = np.load('muons.npy.npz',allow_pickle=True)['sets']
# events = np.reshape(events,(len(events),100,4))[:,:50,:]
# preds = arch.model.predict(events)
# np.savez_compressed("ae_preds_muons.npy", events=events, preds=preds)

infile = open(train_params['outdir']+'trainingHistory.pkl','rb')
history = pickle.load(infile)
if(len(history['val_loss']) == train_params['epochs']):
	val_loss = history['val_loss'][-1]
	val_acc = history['val_accuracy'][-1]
else:
	i = len(history['val_loss']) - train_params['patience_count'] - 1
	val_loss = history['val_loss'][i]
	val_acc = history['val_accuracy'][i]
infile.close()

metrics = {
	"val_loss":val_loss,
	"val_acc":val_acc
}

with open(outdir+"metrics.pkl", 'wb') as f:
    pickle.dump(metrics, f)