import pickle
import os,re
import sys
import time
from decimal import Decimal
import glob
import subprocess
from itertools import product 
import numpy as np
import shutil

def cpuScript():
    f = open('run.sub', 'w')
    submitLines = """
    Universe = vanilla
    +IsLocalJob = true
    Rank = TARGET.IsLocalSlot
    request_disk = 2000MB
    request_memory = 2GB
    request_cpus = 2
    hold = False
    executable              = run_wrapper.sh
    arguments               = {0} $(PROCESS) {2} {3} {4}
    log                     = {2}{0}/log_$(PROCESS).log
    output                  = {2}{0}/out_$(PROCESS).txt
    error                   = {2}{0}/error_$(PROCESS).txt
    should_transfer_files   = Yes
    when_to_transfer_output = ON_EXIT
    transfer_input_files = run_wrapper.sh, fakesNN.py, plotMetrics.py, params.npy, utilities.py, fakeClass.py, validateData.py
    getenv = true
    queue {1}
    """.format(folder, njobs, logDir, repeatSearches, 'false')

    f.write(submitLines)
    f.close()

def gpuScript():
    f = open('run.sub', 'w')
    submitLines = """
    Universe = vanilla
    +IsLocalJob = true
    Rank = TARGET.IsLocalSlot
    request_disk = 2000MB
    request_memory = 2GB
    request_cpus = 2
    hold = False
    executable              = run_wrapper.sh
    arguments               = {0} $(PROCESS) {2} {3} {4}
    log                     = {2}{0}/log_$(PROCESS).log
    output                  = {2}{0}/out_$(PROCESS).txt
    error                   = {2}{0}/error_$(PROCESS).txt
    should_transfer_files   = Yes
    when_to_transfer_output = ON_EXIT
    transfer_input_files = run_wrapper.sh, fakesNN.py, plotMetrics.py, params.npy, utilities.py, fakeClass.py, validateData.py, singularity_wrapper.sh
    getenv = true
    +IsGPUJob = true
    requirements = ((Target.IsGPUSlot == True))
    queue {1}
    """.format(folder, njobs, logDir, repeatSearches, 'true')

    f.write(submitLines)
    f.close()

if __name__=="__main__":

    useGPU = True
    gridSearch = False
    inputDim = 178
    folder = "fakeTracks_4PlusLayer_DYOnly_v1_May25_firstAttempt"
    logDir = "/data/users/mcarrigan/log/disappTrks/fakeTrackNN/Run3/"
    dataDir = ["/store/user/mcarrigan/fakeTracks/converted_DYJets-MC2022_v1/"]
    '''
    delete_elements = ['eventNumber', 'layer1', 'subDet1', 'stripSelection1', 'hitPosX1', 'hitPosY1','layer2', 'subDet2', 'stripSelection2', 'hitPosX2', 'hitPosY2',
                       'layer3', 'subDet3', 'stripSelection3', 'hitPosX3', 'hitPosY3', 'layer4', 'subDet4', 'stripSelection4', 'hitPosX4', 'hitPosY4', 
                       'layer5', 'subDet5', 'stripSelection5', 'hitPosX5', 'hitPosY5', 'layer6', 'subDet6', 'stripSelection6', 'hitPosX6', 'hitPosY6',
                       'layer7', 'subDet7', 'stripSelection7', 'hitPosX7', 'hitPosY7', 'layer8', 'subDet8', 'stripSelection8', 'hitPosX8', 'hitPosY8',
                       'layer9', 'subDet9', 'stripSelection9', 'hitPosX9', 'hitPosY9', 'layer10', 'subDet10', 'stripSelection10', 'hitPosX10', 'hitPosY10', 
                       'layer11', 'subDet11', 'stripSelection11', 'hitPosX11', 'hitPosY11', 'layer12', 'subDet12', 'stripSelection12', 'hitPosX12', 'hitPosY12', 
                       'layer13', 'subDet13', 'stripSelection13', 'hitPosX13', 'hitPosY13', 'layer14', 'subDet14', 'stripSelection14', 'hitPosX14', 'hitPosY14', 
                       'layer15', 'subDet15', 'stripSelection15', 'hitPosX15', 'hitPosY15', 'layer16', 'subDet16', 'stripSelection16', 'hitPosX16', 'hitPosY16']
    '''

    delete_elements = ['passesSelection']

    saveCategories = [{'fake':True, 'real':True, 'pileup':False}, {'fake':True, 'real':False, 'pileup':False}]

    #[filters, batch_norm, undersampling, epochs, dataDir, input_dim, delete_elements, saveCategories, trainPCT, valPCT, loadSplitDataset, dropout]
    #InputDim = 55 (4layers), 64 (5layers), 163 (6+ layers)
    
    params = np.array([[[12, 8], True, -1, 1000, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False, 0.1],
                       [[12, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False, 0.1], 
                       [[12, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False, 0.1],
                       [[12, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False, 0.1],
                       [[12, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False, 0.1],
                       [[12, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False, 0.1],
                       [[12, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False, 0.1],
                       [[12, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False, 0.1],
                       [[12, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False, 0.1],
                       [[12, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False, 0.1]
                       ], dtype='object')

    if not gridSearch:
        np.save('params.npy',np.array(params, dtype='object'))
        njobs = len(params)
        np.save('jobInfo.npy', np.array([1]))

    params = np.load('params.npy', allow_pickle=True)
    jobInfo = np.load('jobInfo.npy', allow_pickle=True)
    repeatSearches = jobInfo[0]
    njobs = len(params)
    if not os.path.exists(logDir + str(folder)): os.mkdir(logDir + str(folder))
    shutil.copy('params.npy', logDir + str(folder))
    shutil.copy('jobInfo.npy', logDir + str(folder))

    if useGPU: gpuScript()
    else: cpuScript()

    os.system('condor_submit run.sub')
