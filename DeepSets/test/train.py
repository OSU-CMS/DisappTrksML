import warnings
warnings.filterwarnings('ignore')

from keras import optimizers, regularizers

from DisappTrksML.DeepSets.deepSetsModel import *
from DisappTrksML.DeepSets.generator import *
from DisappTrksML.DeepSets.utilities import *

model = buildModel()

model.compile(optimizer=optimizers.Adagrad(), 
			  loss='categorical_crossentropy', 
			  metrics=['accuracy'])

params = {
	'dim' : (1000,4),
	'batch_size' : 64,
	'n_classes' : 2,
	'n_channels' : 1,
	'shuffle' : True,
}

file_indexes = {
	'train' : range(0, 750),
	'validation' : range(750, 1000),
	'test' : range(1000, 1414 + 1),
}

training_generator = DataGenerator(file_indexes['train'], **params)
validation_generator = DataGenerator(file_indexes['validation'], **params)

model.fit_generator(
	generator=training_generator,
	validation_data=validation_generator,
	use_multiprocessing=True,
	workers=6)

