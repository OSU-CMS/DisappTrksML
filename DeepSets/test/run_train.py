import os,re
import sys
import time
import numpy as np
import datetime

if __name__=="__main__":

    logdir = "/data/users/llavezzo/Logs/trainingV4/"

    params = [
        [[64, 64, 256],[64, 64, 64],0.5,50],
        [[128,256, 512],[256, 256, 256],0.5,20],
        [[64, 64, 128],[64,32],0.5,50],
        [[64, 64, 256],[64, 64, 64],0.1,20],
        [[128,256, 512],[256, 256, 256],0.1,10],
        [[32, 64, 128],[64,32],0.1,50]
    ]

    np.save('params.npy',params)
    njobs = len(params)

    if(not os.path.isdir(logdir)): os.mkdir(logdir)

    f = open('run.sub', 'w')
    submitLines = """
    Universe = vanilla
    +IsLocalJob = true
    Rank = TARGET.IsLocalSlot
    request_disk = 500MB
    request_memory = 2GB
    request_cpus = 1
    arguments               = $(PROCESS)
    executable              = train_wrapper.sh
    log                     = {0}log_$(PROCESS).log
    output                  = {0}out_$(PROCESS).txt
    error                   = {0}error_$(PROCESS).txt
    when_to_transfer_output = ON_EXIT
    getenv = true
    queue {1}
    """.format(logdir,njobs)

    f.write(submitLines)
    f.close()

    os.system('condor_submit run.sub')
