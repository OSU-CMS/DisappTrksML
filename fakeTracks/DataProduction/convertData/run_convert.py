import pickle
import os,re
import sys
import time
from decimal import Decimal
import glob
import subprocess
import numpy as np

if __name__=="__main__":

    layers = -1
    dataDir = '/store/user/mcarrigan/fakeTracks/selection_v8_NeutrinoGun_ext/'
    #dataDir = '/store/user/mcarrigan/fakeTracks/selection_v7_madgraph_fullSelection/'
    #dataDir = '/store/user/mcarrigan/fakeTracks/selection_higgsino_700_10_v7p1/'
    # v2 means only saving pileup with dR > 0.1 
    outDir = '/store/user/mcarrigan/fakeTracks/converted_NeutrinoGun_ext_cat0p1_4PlusLayer_v8/'
    logDir = '/data/users/mcarrigan/Logs/fakeTracks/converted_NeutrinoGun_ext_cat0p1_4PlusLayer_v8/'
    reprocessAllFiles = True

    if(not os.path.isdir(outDir)): os.mkdir(outDir)
    if(not os.path.isdir(logDir)): os.mkdir(logDir)

    alreadyProcessedFiles = []
    for filename in os.listdir(outDir):
        if('.npz' in filename and 'events' in filename):
            index1 = filename.find("_")
            index2 = filename.find(".")
            numFile = int(filename[index1+1:index2])
            alreadyProcessedFiles.append(numFile)
    files = []
    for filename in os.listdir(dataDir):
        if('.root' in filename and 'hist' in filename):
            index1 = filename.find("_")
            index2 = filename.find(".")
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
    request_disk = 500MB
    request_memory = 2048MB
    request_cpus = 1
    executable              = wrapper.sh
    arguments               = $(PROCESS) {1} {2} {3} {4}
    log                     = {5}log_$(PROCESS).log
    output                  = {5}out_$(PROCESS).txt
    error                   = {5}error_$(PROCESS).txt
    should_transfer_files   = Yes
    when_to_transfer_output = ON_EXIT
    transfer_input_files = {2}, wrapper.sh, convertData.py, Infos.h
    getenv = true
    queue {0}
    """.format(len(files),dataDir,filelist,outDir,layers,logDir)

    f.write(submitLines)
    f.close()

    os.system('condor_submit run.sub')
