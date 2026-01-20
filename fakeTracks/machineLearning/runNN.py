import os
import sys
import numpy as np
import shutil

def cpuScript(container='false'):
    if container=='true':
        containerName='disappTrksCPU.sif'
    else:
        containerName=''

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
    arguments               = {0} $(PROCESS) {2} {3} {4} {5} {6} {8}
    log                     = {2}{0}/log_$(PROCESS).log
    output                  = {2}{0}/out_$(PROCESS).txt
    error                   = {2}{0}/error_$(PROCESS).txt
    should_transfer_files   = Yes
    when_to_transfer_output = ON_EXIT
    +isSmallJob = true
    transfer_input_files = run_wrapper.sh, fakesNN.py, plotMetrics.py, params.npy, utilities.py, fakeClass.py, validateData.py, singularity_wrapper.sh, {5}, {7}
    getenv = true
    queue {1}
    """.format(folder, njobs, logDir, repeatSearches, 'false', config, container, containerName, tune)
    f.write(submitLines)
    f.close()

def gpuScript(container='true'):
    f = open('run.sub', 'w')
    submitLines = """
    Universe = vanilla
    +IsLocalJob = true
    Rank = TARGET.IsLocalSlot
    request_disk = 2000MB
    request_memory = 2GB
    request_cpus = 6
    hold = False
    executable              = run_wrapper.sh
    arguments               = {0} $(PROCESS) {2} {3} {4} {5} {6} {7}
    log                     = {2}{0}/log_$(PROCESS).log
    output                  = {2}{0}/out_$(PROCESS).txt
    error                   = {2}{0}/error_$(PROCESS).txt
    should_transfer_files   = Yes
    when_to_transfer_output = ON_EXIT
    transfer_input_files = run_wrapper.sh, fakesNN.py, plotMetrics.py, params.npy, utilities.py, fakeClass.py, validateData.py, singularity_wrapper.sh, {5}
    getenv = true
    +IsGPUJob = true
    requirements = ((Target.IsGPUSlot == True))
    queue {1}
    """.format(folder, njobs, logDir, repeatSearches, 'true', config, container, tune)
    f.write(submitLines)
    f.close()

if __name__=="__main__":

    useGPU = True
    gridSearch = False
    hpTune = False
    inputDim = 178
    folder = "fakeTracks_oct18-23-noTune"
    logDir = "/data/users/mcarrigan/log/disappTrks/fakeTrackNN/Run3/"

    config = 'test.json'

    container = True

    dataDir = ["/store/user/mcarrigan/fakeTracks/converted_DYJets-MC2022_v1/", "/store/user/mcarrigan/fakeTracks/converted_NeutrinoGun_ext_v9p3/"]
    #dataDir = ["/store/user/mcarrigan/fakeTracks/converted_DYJets_aMCNLO_v9p3/", "/store/user/mcarrigan/fakeTracks/converted_NeutrinoGun_ext_v9p3/"]
    
    delete_elements = ['passesSelection', 'eventNumber', 'layer1', 'subDet1', 'stripSelection1', 'hitPosX1', 'hitPosY1', 'layer2', 'subDet2', 'stripSelection2', 'hitPosX2', 'hitPosY2', 
                       'layer3', 'subDet3', 'stripSelection3', 'hitPosX3', 'hitPosY3', 'layer4', 'subDet4', 'stripSelection4', 'hitPosX4', 'hitPosY4', 
                       'layer5', 'subDet5', 'stripSelection5', 'hitPosX5', 'hitPosY5', 'layer6', 'subDet6', 'stripSelection6', 'hitPosX6', 'hitPosY6', 
                       'layer7', 'subDet7', 'stripSelection7', 'hitPosX7', 'hitPosY7', 'layer8', 'subDet8', 'stripSelection8', 'hitPosX8', 'hitPosY8', 
                       'layer9', 'subDet9', 'stripSelection9', 'hitPosX9', 'hitPosY9', 'layer10', 'subDet10', 'stripSelection10', 'hitPosX10', 'hitPosY10', 
                       'layer11', 'subDet11', 'stripSelection11', 'hitPosX11', 'hitPosY11', 'layer12', 'subDet12', 'stripSelection12', 'hitPosX12', 'hitPosY12', 
                       'layer13', 'subDet13', 'stripSelection13', 'hitPosX13', 'hitPosY13', 'layer14', 'subDet14', 'stripSelection14', 'hitPosX14', 'hitPosY14', 
                       'layer15', 'subDet15', 'stripSelection15', 'hitPosX15', 'hitPosY15', 'layer16', 'subDet16', 'stripSelection16', 'hitPosX16', 'hitPosY16']

    saveCategories = [{'fake':True, 'real':True, 'pileup':False}, {'fake':True, 'real':False, 'pileup':False}]

    #[filters, batch_norm, undersampling, epochs, dataDir, input_dim, delete_elements, saveCategories, trainPCT, valPCT, loadSplitDataset, dropout]
    #InputDim = 55 (4layers), 64 (5layers), 163 (6+ layers)
    
    params = np.array([[[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False, 0.1],
                       [[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False, 0.1], 
                       [[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False, 0.1],
                       [[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False, 0.1],
                       [[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False, 0.1],
                       [[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False, 0.1],
                       [[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False, 0.1],
                       [[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False, 0.1],
                       [[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False, 0.1],
                       [[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False, 0.1]
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

    tune = 'false'
    if hpTune: tune = 'true'

    if useGPU: 
        if not container:
            print("Training on GPU can only be done in singularity container, please set option container=True")
            sys.exit()
        else:
            gpuScript('true')
    else: 
        if not container:
            cpuScript('false')
        else:
            cpuScript('true')

    os.system('condor_submit run.sub')
