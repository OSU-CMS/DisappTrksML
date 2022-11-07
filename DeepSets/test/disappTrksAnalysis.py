#!/usr/bin/env python
import sys

from DisappTrksML.DeepSets.DisappearingTracksAnalysis import *
from DisappTrksML.DeepSets.utilities import *

weights_dirs = {
    'ele_14' : 'train_2022-03-30_16.44.25'
    #'ele_17' : 'lucaWeights/kfold17_noBatchNorm_finalTrainV3',
    #'ele_19' : 'lucaWeights/kfold19_noBatchNorm_finalTrainV3',
    #'muon'   : 'lucaWeights/train_muons',
}

models = { m : ElectronModel() if m.startswith('ele') else MuonModel() for m in weights_dirs }
for m in models:
    models[m].load_model(weights_dirs[m] + '/model.h5')

arch = DeepSetsArchitecture()

payload_dir = os.environ['CMSSW_BASE'] + '/src/OSUT3Analysis/Configuration/data/'

fiducial_maps_2017F = calculateFidicualMaps(
    payload_dir + 'electronFiducialMap_2017_data.root',
    payload_dir + 'muonFiducialMap_2017_data.root',
    '_2017F')

fiducial_maps_mc = calculateFidicualMaps(
    payload_dir + 'electronFiducialMap_2017_data.root',
    payload_dir + 'muonFiducialMap_2017_data.root',
    '')

#####################################

datasets = {
    'electrons' : ['/store/user/bfrancis/images_v7/SingleEle_2017F_wIso/0000/images_*.root', '/store/user/bfrancis/images_v7/SingleEle_2017F_wIso/0001/images_*.root'],
    'muons'     : ['/store/user/bfrancis/images_v7/SingleMuo_2017F_wIso/0000/images_*.root', '/store/user/bfrancis/images_v7/SingleMuo_2017F_wIso/0001/images_*.root'],
    'fake'      : ['/store/user/bfrancis/images_v7/ZtoMuMu_2017F_noIsoCut/0000/images_*.root', '/store/user/bfrancis/images_v7/ZtoMuMu_2017F_noIsoCut/0001/images_*.root'],
}

datasets.update(
    { 'higgsino_%d_%d' % (mass, lifetime) : ['/store/user/mcarrigan/AMSB/images_v7/images_higgsino_%dGeV_%dcm_step3/hist_*' % (mass, lifetime)] for mass in [700] for lifetime in [10, 100, 1000, 10000] }
)

#####################################
'''
if len(sys.argv) > 2:
    dataset = str(sys.argv[1])
    fileNumber = int(sys.argv[2])

    inputFiles = []
    for inputDir in datasets[dataset]:
        inputFiles.extend(glob.glob(inputDir))

    n, results = processFile(
        inputFiles[fileNumber],
        models, arch,
        fiducial_maps_mc if dataset.startswith('higgsino_') else fiducial_maps_2017F,
        dataset)
    if n > 0:
        np.savez_compressed('output_' + dataset + '_' + str(fileNumber) + '.npz', results=np.array(results))
elif len(sys.argv) > 1:
    dataset = str(sys.argv[1])
    processDataset(dataset, datasets[dataset], models, arch, fiducial_maps_mc if dataset.startswith('higgsino_') else fiducial_maps_2017F)
else:
    for dataset in datasets:
        processDataset(dataset, datasets[dataset], models, arch, fiducial_maps_mc if dataset.startswith('higgsino_') else fiducial_maps_2017F)
'''
analyze(datasets, 'outputFiles')
