import pickle
import os,re
import sys
import time
from decimal import Decimal
import glob
import subprocess
import numpy as np

if __name__=="__main__":

    dataDir = '/store/user/mcarrigan/Images-v9-DYJets-MC2017_aMCNLO_ext/'
    #dataDir = '/store/user/mcarrigan/AMSB/images_v7/images_higgsino_700GeV_10000cm_step3/'
    outDir = '/store/user/mcarrigan/deepSets/electronModel/'
    logDir = '/data/users/mcarrigan/Logs/deepSets/electronModel/'
    reprocessAllFiles = True

    if(not os.path.isdir(outDir)): os.mkdir(outDir)
    if(not os.path.isdir(logDir)): os.mkdir(logDir)

    alreadyProcessedFiles = []
    for filename in os.listdir(outDir):
        if('.root' in filename and 'images' in filename):
            index1 = filename.find("_")
            index2 = filename.find(".root")
            numFile = int(filename[index1+1:index2])
            alreadyProcessedFiles.append(numFile)
    files = []
    for filename in os.listdir(dataDir):
        if('.root' in filename and 'images' in filename):
            index1 = filename.find("_")
            index2 = filename.find(".root")
            numFile = int(filename[index1+1:index2])
            if(not reprocessAllFiles):
                if(numFile in alreadyProcessedFiles): continue
            files.append(numFile) 
    filelist = 'filelist.txt'
    np.savetxt(filelist,files)

    f = open('run.sub', 'w')
    submitLines = """
    Universe = vanilla
    +IsLocalJob = true
    Rank = TARGET.IsLocalSlot
    request_disk = 100MB
    request_memory = 1GB
    request_cpus = 2
    executable              = convert_wrapper.sh
    arguments               = $(PROCESS) {1} {2} {3}
    log                     = {4}log_$(PROCESS).log
    output                  = {4}out_$(PROCESS).txt
    error                   = {4}error_$(PROCESS).txt
    when_to_transfer_output = ON_EXIT
    getenv = true
    queue {0}
    """.format(len(files),filelist,dataDir,outDir, logDir)

    f.write(submitLines)
    f.close()

    os.system('condor_submit run.sub')
