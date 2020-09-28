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

    folder = "mlp_9_15"
    dataDir = "/store/user/mcarrigan/disappearingTracks/electron_selection_tanh_V3/"
    params = [

        #undersampling
        [[64,128,64],False,-1,5,dataDir], [[64,128],False,-1,5,dataDir], [[32, 64, 32],False,-1,5,dataDir]

    ]
    np.save('params_mlp.npy',params)
    njobs = len(params)

    os.system('mkdir /data/users/mcarrigan/Logs/'+str(folder))

    f = open('run.sub', 'w')
    submitLines = """

    Universe = vanilla
    +IsLocalJob = true
    Rank = TARGET.IsLocalSlot
    request_disk = 250MB
    request_memory = 2GB
    request_cpus = 3
    executable              = run_wrapper_mlp.sh
    arguments               = {0} $(PROCESS)
    log                     = /data/users/mcarrigan/Logs/{0}/log_$(PROCESS).log
    output                  = /data/users/mcarrigan/Logs/{0}/out_$(PROCESS).txt
    error                   = /data/users/mcarrigan/Logs/{0}/error_$(PROCESS).txt
    should_transfer_files   = Yes
    when_to_transfer_output = ON_EXIT
    transfer_input_files = run_wrapper_mlp.sh, flow_mlp.py, utils.py, validate_mlp.py, params_mlp.npy
    getenv = true
    queue {1}

    """.format(folder, njobs)

    f.write(submitLines)
    f.close()

    os.system('condor_submit run.sub')
