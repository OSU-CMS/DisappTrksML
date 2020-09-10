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

    folder = "deepSets100_3"
    dataDir = '/store/user/llavezzo/disappearingTracks/converted_deepSets100_failAllRecos/'
    params = [
        #undersampling
        [0.9,20,dataDir],       
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
    request_cpus = 4
    executable              = run_wrapper.sh
    arguments               = {0} $(PROCESS)
    log                     = /data/users/llavezzo/Logs/{0}/log_$(PROCESS).log
    output                  = /data/users/llavezzo/Logs/{0}/out_$(PROCESS).txt
    error                   = /data/users/llavezzo/Logs/{0}/error_$(PROCESS).txt
    should_transfer_files   = Yes
    when_to_transfer_output = ON_EXIT
    transfer_input_files = run_wrapper.sh, train.py, utils.py, generator.py, validate.py, params.npy, model.py
    getenv = true
    queue {1}

    """.format(folder, njobs)

    f.write(submitLines)
    f.close()

    os.system('condor_submit run.sub')