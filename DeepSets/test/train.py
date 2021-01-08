import warnings
warnings.filterwarnings('ignore')
import glob, os

import tensorflow as tf
from sklearn.model_selection import KFold

from DisappTrksML.DeepSets.architecture import *
from DisappTrksML.DeepSets.generator import *
from DisappTrksML.DeepSets.utilities import *

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
	outdir = input_params[4]

model_params = {
	'phi_layers':input_params[0],
	'f_layers':input_params[1]
}
val_generator_params = {
	'input_dir' : '/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_converted/',
	'batch_size' : 256,
	'with_info' : False,
	'maxHits' : 100
}
train_generator_params = val_generator_params.copy()
train_generator_params.update({
	'shuffle': True,
	'batch_ratio': input_params[2]
})
train_params = {
	'epochs':input_params[3],
	'outdir':outdir,
	'patience_count':5
}

if(not os.path.isdir(outdir)): os.mkdir(outdir)

arch = DeepSetsArchitecture(**model_params)
arch.buildModel()

inputFiles = glob.glob(train_generator_params['input_dir']+'images_*.root.npz')
inputIndices = np.array([f.split('images_')[-1][:-9] for f in inputFiles])
nFiles = len(inputIndices)
print('Found', nFiles, 'input files')

file_ids = {
	'train'      : inputIndices[:500],
	'validation' : inputIndices[500:600]
}

train_generator = DataGeneratorV3(file_ids['train'], **train_generator_params)
val_generator = DataGeneratorV4(file_ids['validation'], **val_generator_params)

arch.fit_generator(train_generator=train_generator, 
				   val_generator=val_generator, 	
					**train_params)

arch.save_trainingHistory(train_params['outdir']+'trainingHistory.pkl')
arch.plot_trainingHistory(train_params['outdir']+'trainingHistory.pkl',train_params['outdir']+'trainingHistory.png','loss')
arch.save_weights(train_params['outdir']+'model_weights.h5')
arch.save_model(train_params['outdir']+'model.h5')
arch.save_metrics(train_params['outdir']+'trainingHistory.pkl',train_params['outdir']+"metrics.pkl", train_params)