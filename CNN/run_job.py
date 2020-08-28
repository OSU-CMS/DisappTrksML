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

    folder = "failAllRecos"
    dataDir = "/store/user/llavezzo/disappearingTracks/electron_selection_failAllRecos/"
    params = [

        #undersampling
        [[128,256,512],False,0.5,10,dataDir],        #varying layers/filters
        [[256,512,512],False,0.8,10,dataDir],

        #no undersampling, weights
        [[256,512],True,-1,5,dataDir],          #varying layers/filters

    ]
    np.save('params.npy',params)
    njobs = len(params)

    os.system('mkdir /data/users/llavezzo/Logs/'+str(folder))

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
    log                     = /data/users/llavezzo/Logs/{0}/log_$(PROCESS).log
    output                  = /data/users/llavezzo/Logs/{0}/out_$(PROCESS).txt
    error                   = /data/users/llavezzo/Logs/{0}/error_$(PROCESS).txt
    should_transfer_files   = Yes
    when_to_transfer_output = ON_EXIT
    transfer_input_files = run_wrapper.sh, flow.py, utils.py, validate.py, params.npy
    getenv = true
    queue {1}

    """.format(folder, njobs)

    f.write(submitLines)
    f.close()

    os.system('condor_submit run.sub')