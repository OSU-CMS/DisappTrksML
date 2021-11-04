import pickle
import os,re
import sys
import time
from decimal import Decimal
import glob
import subprocess
from itertools import product 
import numpy as np

if __name__=="__main__":


    inputDim = 178
    folder = "fakeTracks_4PlusLayer_aMCv9p3_11_4_NGBoost_filterSearch"
    logDir = "/data/users/mcarrigan/fakeTracks/networks/filterSearch/"
    dataDir = ["/store/user/mcarrigan/fakeTracks/converted_DYJets_aMCNLO_v9p3/", "/store/user/mcarrigan/fakeTracks/converted_NeutrinoGun_ext_v9p3/"]
    #delete_elements = ['dEdxPixel', 'pixelHitSize7', 'charge8', 'hitPosY9', 'numSatMeasurements', 'layer13', 'pixelHitSizeX12', 'pixelHitSize15', 'pixelHitSize13']
    delete_elements = ['eventNumber', 'layer1', 'subDet1', 'stripSelection1', 'hitPosX1', 'hitPosY1','layer2', 'subDet2', 'stripSelection2', 'hitPosX2', 'hitPosY2',
                       'layer3', 'subDet3', 'stripSelection3', 'hitPosX3', 'hitPosY3', 'layer4', 'subDet4', 'stripSelection4', 'hitPosX4', 'hitPosY4', 
                       'layer5', 'subDet5', 'stripSelection5', 'hitPosX5', 'hitPosY5', 'layer6', 'subDet6', 'stripSelection6', 'hitPosX6', 'hitPosY6',
                       'layer7', 'subDet7', 'stripSelection7', 'hitPosX7', 'hitPosY7', 'layer8', 'subDet8', 'stripSelection8', 'hitPosX8', 'hitPosY8',
                       'layer9', 'subDet9', 'stripSelection9', 'hitPosX9', 'hitPosY9', 'layer10', 'subDet10', 'stripSelection10', 'hitPosX10', 'hitPosY10', 
                       'layer11', 'subDet11', 'stripSelection11', 'hitPosX11', 'hitPosY11', 'layer12', 'subDet12', 'stripSelection12', 'hitPosX12', 'hitPosY12', 
                       'layer13', 'subDet13', 'stripSelection13', 'hitPosX13', 'hitPosY13', 'layer14', 'subDet14', 'stripSelection14', 'hitPosX14', 'hitPosY14', 
                       'layer15', 'subDet15', 'stripSelection15', 'hitPosX15', 'hitPosY15', 'layer16', 'subDet16', 'stripSelection16', 'hitPosX16', 'hitPosY16']
    saveCategories = [{'fake':True, 'real':True, 'pileup':False}, {'fake':True, 'real':False, 'pileup':False}]
    #[filters, batch_norm, undersampling, epochs, dataDir, input_dim, input variables to delete, types of track to use, %of data to train on, % of (1-train) data to validate on]
    #InputDim = 55 (4layers), 64 (5layers), 163 (6+ layers)

    #params = np.array([[[32, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5],
    #          [[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5], 
    #          [[24, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5],
    #          [[24, 12], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5]], dtype='object')

    
    params = np.array([[[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False],
              [[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False], 
              [[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False],
              [[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False],
              [[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False],
              [[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False],
              [[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False],
              [[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False],
              [[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False],
              [[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5, False]], dtype='object')

    #np.save('params.npy',np.array(params, dtype='object'))
    #njobs = len(params)
    

    params = np.load('params.npy', allow_pickle=True)
    njobs = len(params)
    os.system('mkdir ' + logDir + str(folder))

    f = open('run.sub', 'w')
    submitLines = """
    Universe = vanilla
    +IsLocalJob = true
    Rank = TARGET.IsLocalSlot
    request_disk = 250MB
    request_memory = 2GB
    request_cpus = 3
    executable              = run_wrapper.sh
    arguments               = {0} $(PROCESS) {2}
    log                     = {2}{0}/log_$(PROCESS).log
    output                  = {2}{0}/out_$(PROCESS).txt
    error                   = {2}{0}/error_$(PROCESS).txt
    should_transfer_files   = Yes
    when_to_transfer_output = ON_EXIT
    transfer_input_files = run_wrapper.sh, fakesNN.py, plotMetrics.py, params.npy, utilities.py, fakeClass.py
    getenv = true
    queue {1}
    """.format(folder, njobs, logDir)

    f.write(submitLines)
    f.close()

    os.system('condor_submit run.sub')
