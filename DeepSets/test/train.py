import warnings
warnings.filterwarnings('ignore')

import glob

from DisappTrksML.DeepSets.architecture import *
from DisappTrksML.DeepSets.generator import *
from DisappTrksML.DeepSets.utilities import *

arch = DeepSetsArchitecture()

arch.buildModel()

params = {
	'inputDir' : '/store/user/bfrancis/numpy/electrons_DYJetsToLL/',
	'dim' : (100,4),
	'batch_size' : 64,
	'n_classes' : 2,
	'shuffle' : True,
}

inputFiles = glob.glob('/store/user/bfrancis/numpy/electrons_DYJetsToLL/hist_*.root.npz')
inputIndices = [f.split('hist_')[-1][:-9] for f in inputFiles]
nFiles = len(inputIndices)

print('Found', nFiles, 'input files')

file_indexes = {
	'train' : inputIndices[0 : nFiles * 3/5],
	'validation' : inputIndices[nFiles * 3/5 : nFiles * 4/5],
	'test' : inputIndices[nFiles * 4/5 : -1],
}

train_generator = DataGenerator(file_indexes['train'], **params)
validation_data = DataGenerator(file_indexes['validation'], **params)

arch.fit_generator(train_generator=train_generator, validation_data=validation_data)

arch.displayTrainingHistory()
