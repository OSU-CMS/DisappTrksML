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
	outdir = "trainV4_param"+str(sys.argv[1])+"/"
	input_params = np.load("params.npy",allow_pickle=True)[int(sys.argv[1])]
n_splits = 5

model_params = {
	'phi_layers':input_params[0],
	'f_layers':input_params[1],
	'track_info_shape':3
}
generator_params = {
	'input_dir' : '/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_converted/',
	'batch_size' : 128,
	'batch_ratio' : input_params[2],
	'shuffle' : True,
}
train_params = {
	'epochs':input_params[3],
	'outdir':outdir,
	'monitor':'val_loss',
	'patience_count':3
}


if(not os.path.isdir(outdir)): os.mkdir(outdir)

arch = DeepSetsArchitecture(**model_params)
arch.buildModel()

inputFiles = glob.glob(generator_params['input_dir']+'images_*.root.npz')
inputIndices = np.array([f.split('images_')[-1][:-9] for f in inputFiles])
nFiles = len(inputIndices)
print('Found', nFiles, 'input files')

kf = KFold(n_splits=n_splits,random_state=42,shuffle=True)
k, val_loss, val_acc = 0,0,0

for train_index, test_index in kf.split(inputIndices):

	file_ids = {
		'train'      : inputIndices[train_index],
		'validation' : inputIndices[test_index]
	}

	train_generator = DataGeneratorV3(file_ids['train'], **generator_params)
	val_generator = DataGeneratorV3(file_ids['validation'], **generator_params)

	arch.fit_generator(train_generator=train_generator, 
						val_generator=val_generator, 	
						**train_params)

	arch.save_trainingHistory(train_params['outdir']+'trainingHistory_'+str(k)+'.pkl')

	infile = open(outdir+"trainingHistory_"+str(k)+".pkl",'rb')
	history = pickle.load(infile)

	if(len(history['val_loss']) == train_params['epochs']):
		val_loss += history['val_loss'][-1]
		val_acc += history['val_accuracy'][-1]
	else:
		i = train_params['epochs'] - train_params['patience_count'] - 1
		val_loss += history['val_loss'][i]
		val_acc += history['val_accuracy'][i]
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