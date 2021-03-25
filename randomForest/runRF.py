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

    folder = "fakeTracks_randomForest_3_19"
    logDir = "/data/users/mcarrigan/Logs/fakeTracks/"
    #dataDir = ["/store/user/mcarrigan/fakeTracks/converted_madgraph_4PlusLayer_v7p1/", "/store/user/mcarrigan/fakeTracks/converted_aMC_4PlusLayer_v7p1/"]
    dataDir = ["/store/user/mcarrigan/fakeTracks/converted_madgraph_5layer_v7p1/", "/store/user/mcarrigan/fakeTracks/converted_aMC_5layer_v7p1/"]
    #[filters, batch_norm, undersampling, epochs, dataDir, input_dim]
    #InputDim = 55 (4layers), 64 (5layers), 163 (6+ layers)
    #params = [[[16, 4], True, 0.9, 100, dataDir, 64],
    #          [[16, 4], True, 0.8, 100, dataDir, 64], 
    #          [[16, 4], True, 0.6, 100, dataDir, 64],
    #          [[16, 4], True, 0.4, 100, dataDir, 64]]
    #np.save('params.npy',params)
    njobs = 1

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
    transfer_input_files = run_wrapper.sh, fakesRF.py
    getenv = true
    queue {1}
    """.format(folder, njobs, logDir)

    f.write(submitLines)
    f.close()

    os.system('condor_submit run.sub')
