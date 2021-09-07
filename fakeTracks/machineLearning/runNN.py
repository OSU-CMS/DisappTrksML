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

    inputDim = 173
    folder = "fakeTracks_4PlusLayer_aMCv9p1_9_7_NGBoost_test1"
    logDir = "/data/users/mcarrigan/Logs/fakeTracks/"
    #dataDir = ["/store/user/mcarrigan/fakeTracks/converted_madgraph_4PlusLayer_v7p1/", "/store/user/mcarrigan/fakeTracks/converted_aMC_4PlusLayer_v7p1/"]
    #dataDir = ["/store/user/mcarrigan/fakeTracks/converted_DYJets_aMCNLO_PUTrain_v9p1/", "/store/user/mcarrigan/fakeTracks/converted_NeutrinoGun_ext_4PlusLayer_v9p1/"]
    dataDir = ["/store/user/mcarrigan/fakeTracks/converted_v9_DYJets_aMCNLO_4PlusLayer_v9p1/", "/store/user/mcarrigan/fakeTracks/converted_NeutrinoGun_ext_4PlusLayer_v9p1/"]
    delete_elements = ['totalCharge', 'numSatMeasurements', 'stripSelection', 'hitPosX', 'hitPosY']
    saveCategories = [{'fake':True, 'real':True, 'pileup':False}, {'fake':True, 'real':False, 'pileup':False}]
    #[filters, batch_norm, undersampling, epochs, dataDir, input_dim, input variables to delete, types of track to use, %of data to train on, % of (1-train) data to validate on]
    #InputDim = 55 (4layers), 64 (5layers), 163 (6+ layers)

    params = np.array([[[32, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5],
              [[16, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5], 
              [[24, 8], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5],
              [[24, 12], True, -1, 100, dataDir, inputDim, delete_elements, saveCategories, 0.7, 0.5]], dtype='object')
    np.save('params.npy',np.array(params, dtype='object'))
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
    arguments               = {0} $(PROCESS)
    log                     = {2}{0}/log_$(PROCESS).log
    output                  = {2}{0}/out_$(PROCESS).txt
    error                   = {2}{0}/error_$(PROCESS).txt
    should_transfer_files   = Yes
    when_to_transfer_output = ON_EXIT
    transfer_input_files = run_wrapper.sh, fakesNN.py, plotMetrics.py, params.npy, utilities.py
    getenv = true
    queue {1}
    """.format(folder, njobs, logDir)

    f.write(submitLines)
    f.close()

    os.system('condor_submit run.sub')
