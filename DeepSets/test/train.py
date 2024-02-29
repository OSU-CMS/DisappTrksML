import warnings
warnings.filterwarnings('ignore')
import glob, os

import tensorflow as tf
from datetime import datetime
import numpy as np
import sys
from DisappTrksML.DeepSets.ElectronModel import ElectronModel
from DisappTrksML.DeepSets.generator import BalancedGenerator, Generator
#from DisappTrksML.DeepSets.utilities import *

if False:
        # limit CPU usage
        config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads = 4,   
                                                                        intra_op_parallelism_threads = 4,
                                                                        allow_soft_placement = True,
                                                                        device_count={'CPU': 4})
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

#######
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
backup_suffix = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
outdir = "train_"+backup_suffix+"/"
print("output directory", outdir)

info_indices = [4,      # nPV
                                8,      # eta
                                9,      # phi
                                12      # nValidPixelHits
                                ]
model_params = {
        'eta_range':0.25,
        'phi_range':0.25,
        'phi_layers':[64,64,256],
        'f_layers':[64,64,64],
        'max_hits' : 100,
        'track_info_indices' : info_indices
}       
val_generator_params = {
        'input_dir' : '/store/user/rsantos/2022/combined_DYJet/training/',
        'batch_size' : 256,
        'max_hits' : 100,
        'info_indices' : info_indices
}
train_generator_params = val_generator_params.copy()
train_generator_params.update({
        'shuffle': True,
        'batch_ratio': 0.5
})
train_params = {
        'epochs': 1,
        'outdir':outdir,
        'patience_count':20
}

if(len(sys.argv)>1):
        input_params = np.load("params.npy",allow_pickle=True)[int(sys.argv[1])]
        print(input_params)
        train_params['outdir'] = str(input_params[-1])
        model_params['phi_layers'] = input_params[0]
        model_params['f_layers'] = input_params[1]


if(not os.path.isdir(train_params["outdir"])): os.mkdir(train_params["outdir"])

arch = ElectronModel(**model_params)
arch.build_model()

inputFiles = glob.glob(train_generator_params['input_dir']+'*.root.npz')
inputIndices = np.array([f.split('images_')[-1][:-9] for f in inputFiles])
nFiles = len(inputIndices)
print(('Found', nFiles, 'input files'))

file_ids = {
        'train'      : inputIndices[:int(nFiles*0.9)],
        'validation' : inputIndices[int(nFiles*0.9):]
}

train_generator = BalancedGenerator(file_ids['train'], **train_generator_params)
val_generator = Generator(file_ids['validation'], **val_generator_params)

arch.train_model(data_directory= train_generator_params['input_dir'],
                 epochs = 1,
                 patience_count = 20,
                 outdir = train_params['outdir'],
                 val_generator_params = val_generator_params,
                 train_generator_params = train_generator_params)

arch.saveGraph(train_params['outdir'])
arch.saveWeights

#arch.save_model(train_params['outdir']+'model.h5')
