import os,re
import sys
import time
import numpy as np
import datetime

if __name__=="__main__":

    logdir = "/data/users/rsantos/condor/DeepSets"
    datadir = "/data/users/rsantos/condor/DeepSets"

    params = [
        [[16,16,16],[16,16,16],datadir+"model0/"],
        # [[16,16,16],[32,32,32],datadir+"model/"],
        # [[16,16,16],[64,64,64],datadir+"model2/"],
        # [[16,16,16],[128,128,128],datadir+"model3/"],
        # [[16,16,16],[256,256,256],datadir+"model4/"],
        # [[16,16,16],[16, 32, 16],datadir+"model5/"],
        # [[16,16,16],[64, 32, 16],datadir+"model6/"],
        # [[16,16,16],[128, 64, 32],datadir+"model7/"],
        # [[32,32,32],[16,16,16],datadir+"model8/"],
        # [[32,32,32],[32,32,32],datadir+"model9/"],
        # [[32,32,32],[64,64,64],datadir+"model10/"],
        # [[32,32,32],[128,128,128],datadir+"model11/"],
        # [[32,32,32],[256,256,256],datadir+"model12/"],
        # [[32,32,32],[16, 32, 16],datadir+"model13/"],
        # [[32,32,32],[16, 64, 16],datadir+"model14/"],
        # [[32,32,32],[16, 128, 16],datadir+"model15/"],
        # [[32,32,32],[16, 256, 16],datadir+"model16/"],
        # [[32,32,32],[16, 16, 16],datadir+"model17/"],
        # [[32,32,32],[64, 32, 16],datadir+"model18/"],
        # [[32,32,32],[128, 64, 32],datadir+"model19/"],
        # [[64,32,16],[16,16,16],datadir+"model20/"],
        # [[64,32,16],[32,32,32],datadir+"model21/"],
        # [[64,32,16],[64,64,64],datadir+"model22/"],
        # [[64,32,16],[128,128,128],datadir+"mode23/"],
        # [[64,32,16],[256,256,256],datadir+"model24/"],
        # [[64,32,16],[16, 32, 16],datadir+"model25/"],
        # [[64,32,16],[16, 64, 16],datadir+"model26/"],
        # [[64,32,16],[16, 128, 16],datadir+"model27/"],
        # [[64,32,16],[16, 256, 16],datadir+"model28/"],
        # [[64,32,16],[16, 16, 16],datadir+"model29/"],
        # [[64,32,16],[64, 32, 16],datadir+"model30/"],
        # [[64,32,16],[128, 64, 32],datadir+"model31/"],
        # [[64,32],[128, 64],datadir+"model32/"],
        # [[32,16],[32, 16],datadir+"model33/"],

    ]

    np.save('params.npy',params)
    njobs = len(params)

    if(not os.path.isdir(logdir)): os.mkdir(logdir)
    if(not os.path.isdir(datadir)): os.mkdir(datadir)

    f = open('run.sub', 'w')
    submitLines = """
    Universe = vanilla
    +IsLocalJob = true
    Rank = TARGET.IsLocalSlot
    request_disk = 1GB
    request_memory = 3GB
    request_cpus = 6
    hold = False
    transfer_input_files = train_wrapper.sh 
    arguments               = $(PROCESS)
    executable              = train_wrapper.sh
    log                     = {0}log_$(PROCESS).log
    output                  = {0}out_$(PROCESS).txt
    error                   = {0}error_$(PROCESS).txt
    should_transfer_files = Yes
    when_to_transfer_output = ON_EXIT
    getenv = true
    +IsGPUJob = true
    requirements = ((Target.IsGPUSlot == True))
    queue {1}
    """.format(logdir,njobs)

    f.write(submitLines)
    f.close()

    os.system('condor_submit run.sub')
