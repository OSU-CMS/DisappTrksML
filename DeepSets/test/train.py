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
n_splits = 5

params = {
	'input_dir' : '/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_converted/',
	'dim' : (100, 4),
	'batch_size' : 128,
	'batch_ratio' : 0.5,
	'n_classes' : 2,
	'shuffle' : True,
}

if(not os.path.isdir(outdir)): os.mkdir(outdir)

arch = DeepSetsArchitecture()
arch.buildModel()

inputFiles = glob.glob(params['input_dir']+'hist_*.root.npz')
inputIndices = np.array([f.split('hist_')[-1][:-9] for f in inputFiles])
nFiles = len(inputIndices)
print('Found', nFiles, 'input files')

kf = KFold(n_splits=n_splits,random_state=42,shuffle=True)
k, val_loss, val_acc = 0,0,0

for train_index, test_index in kf.split(inputIndices):

	file_ids = {
		'train'      : inputIndices[train_index],
		'validation' : inputIndices[test_index]
		# 'test'       : inputIndices[nFiles * 4/5 : -1],
	}

	train_generator = DataGeneratorV3(file_ids['train'], **params)
	validation_data = DataGeneratorV3(file_ids['validation'], **params)

	arch.fit_generator(train_generator=train_generator, 
						validation_data=validation_data, 	
						epochs=1,
						outdir=outdir)

	arch.save_weights(outdir+'model_'+str(k)+'.h5')
	arch.save_trainingHistory(outdir+'trainingHistory_'+str(k)+'.pkl')

	infile = open(outdir+"trainingHistory_"+str(k)+".pkl",'rb')
	history = pickle.load(infile)

	val_loss += history['val_loss'][-1]
	val_acc += history['val_accuracy'][-1]
	infile.close()

	k+=1

val_loss /= k
val_acc /= k

metrics = {
	"val_loss":val_loss,
	"val_acc":val_acc
}

with open(outdir+"metrics.pkl", 'wb') as f:
    pickle.dump(metrics, f)