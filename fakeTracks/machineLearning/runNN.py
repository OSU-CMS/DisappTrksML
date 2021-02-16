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

    folder = "fakeTracks_2_15"
    logDir = "/data/users/mcarrigan/Logs/"
    dataDir = ["/store/user/mcarrigan/fakeTracks/converted_v1/", "/store/user/mcarrigan/fakeTracks/converted_aMC_v1/"]
    #[filters, class_weights, undersampling, epochs, dataDir]
    params = [[[12, 8], False, -1, 100, dataDir], 
              [[64, 32, 12, 6], False, -1, 100, dataDir], 
              [[128, 64, 24, 6], False, -1, 100, dataDir], 
              [[512, 256, 128, 64, 32, 16,8], False, -1, 100, dataDir]]
    np.save('params.npy',params)
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
    transfer_input_files = run_wrapper.sh, fakesNN.py, plotMetrics.py, params.npy
    getenv = true
    queue {1}
    """.format(folder, njobs, logDir)

    f.write(submitLines)
    f.close()

    os.system('condor_submit run.sub')
