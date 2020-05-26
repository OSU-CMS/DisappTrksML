import utils
import numpy as np

dataDir = '/home/MilliQan/data/disappearingTracks/tracks/'
workDir = '/home/llavezzo/'
plotDir = workDir + 'plots/'

data_e = np.load(dataDir+'e_DYJets50V3_norm_20x20.npy')
data_bkg = np.load(dataDir+'bkg_DYJets50V3_norm_20x20.npy')

utils.save_event(data_e[0,:,:],plotDir,'sample_event')