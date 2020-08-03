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

    folder = "undersample_study"
    params = [
        [False, 0.7, 20],
        [False, 0.9, 20],
    ]
    np.save('params',params)
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
    transfer_input_files = {1}run_wrapper.sh, {1}flow.py, {1}utils.py, {1}validate.py, {1}params.npy
    getenv = true
    queue {2}

    """.format(folder, "/share/scratch0/llavezzo/CMSSW_11_1_2_patch1/src/DisappTrksML/CNN/", njobs)

    f.write(submitLines)
    f.close()

    os.system('condor_submit run.sub')