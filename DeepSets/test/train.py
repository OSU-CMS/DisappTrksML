import warnings
warnings.filterwarnings('ignore')

import glob

from DisappTrksML.DeepSets.architecture import *
from DisappTrksML.DeepSets.generator import *
from DisappTrksML.DeepSets.utilities import *

import tensorflow as tf

if False:
	# limit CPU usage
	config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads = 4,   
									intra_op_parallelism_threads = 4,
									allow_soft_placement = True,
									device_count={'CPU': 4})
	tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

#######

arch = DeepSetsArchitecture()

arch.buildModel()

params = {
	'input_dir' : '/store/user/llavezzo/disappearingTracks/images_v5_DYJetsToll_muons/',
	'dim' : (100, 4),
	'batch_size' : 32,
	'batch_ratio' : 0.5,
	'n_classes' : 2,
	'shuffle' : True,
}

inputFiles = glob.glob(params['input_dir']+'hist_*.root.npz')
inputIndices = [f.split('hist_')[-1][:-9] for f in inputFiles]
nFiles = len(inputIndices)

print('Found', nFiles, 'input files')

file_ids = {
	'train'      : inputIndices[0 : nFiles * 3/5],
	'validation' : inputIndices[nFiles * 4/5 : nFiles]
	# 'test'       : inputIndices[nFiles * 4/5 : -1],
}

train_generator = DataGeneratorV3(file_ids['train'], **params)
validation_data = DataGeneratorV3(file_ids['validation'], **params)

arch.fit_generator(train_generator=train_generator, validation_data=validation_data, epochs=2)

arch.displayTrainingHistory()