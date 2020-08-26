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

    folder = "cnn_debug"
    dataDir = "/store/user/llavezzo/disappearingTracks/electron_selection_failSelection/"
    params = [
        [[256,512,1024],False,0.8,2,dataDir],
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