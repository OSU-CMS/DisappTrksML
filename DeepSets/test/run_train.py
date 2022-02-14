import os,re
import sys
import time
import numpy as np
import datetime

if __name__=="__main__":

    logdir = "/data/users/mcarrigan/Logs/train/"
    datadir = "/data/users/mcarrigan/train/"

    params = [
        [[64,64,64,64,64,64,128],[128,64],0.1,30,datadir+"kfold0/"]
        # [[64,64,64,64,64,64,128],[128,64],0.01,15,"kfold1/"],
        # [[64,64,128],[128,64,64,64,64,64],0.1,30,"kfold2/"],
        # [[64,64,128],[128,64,64,64,64,64],0.01,15,"kfold3/"],
        # [[64,32],[32,32],0.1,50,"kfold4/"],
        # [[64,32],[32,32],0.01,50,"kfold5/"],
        # [[256,128],[128,64,32,32,32],0.1,30,"kfold6/"],
        # [[256,128],[128,64,32,32,32],0.01,15,"kfold7/"],
        # [[64,64,256],[64,64,64],0.1,40,"kfold8/"],
        # [[64,64,256],[64,64,64],0.01,20,"kfold9/"],
        # [[64,64,256],[64,64,64],0.01,30,"kfold10_noBatchNorm/", False],
        # [[400,256,128], [128,128,64,32],0.1,30,"kfold11_noBatchNorm/", False],
        # [[400,256,128], [128,128,64,32],0.01,30,"kfold12_noBatchNorm/", False],
        # [[400,128],[128,64,32,32,32,32,32],0.01,30,"kfold14_noBatchNorm/", False],
        # [[256,512,256], [128,64,32],0.01,30,"kfold15_noBatchNorm/", False],
        # [[64,64,256],[64,64,64],0.5,50,"kfold16_noBatchNorm/", True],
        # [[64,64,256],[64,64,64],0.1,30,"kfold17_noBatchNorm/", True],
        # [[64,64,256],[64,64,64],0.01,15,"kfold18_noBatchNorm/", True],
        # [[400,256,128], [128,128,64,32],0.5,50,"kfold19_noBatchNorm/", True],
        # [[400,256,128], [128,128,64,32],0.1,30,"kfold20_noBatchNorm/", True],
        # [[400,256,128], [128,128,64,32],0.01,15,"kfold21_noBatchNorm/", True],
        # [[400,128],[128,64],0.5,100,"kfold22_noBatchNorm/", True],
        # [[400,128],[128,64],0.1,40,"kfold23_noBatchNorm/", True],
        # [[400,128],[128,64],0.01,20,"kfold24_noBatchNorm/", True]

        # [[400,128],[128,64,32,32,32,32,32],0.01,25,"kfold14_noBatchNorm_finalTrainV3/", True],
        # [[400,256,128], [128,128,64,32],0.5,15,"kfold19_noBatchNorm_finalTrainV3/", True],
        # [[64,64,256],[64,64,64],0.1,20,"kfold17_noBatchNorm_finalTrainV3/", True],

        #["train_output_4/"]
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
    request_disk = 500MB
    request_memory = 2GB
    request_cpus = 4
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
