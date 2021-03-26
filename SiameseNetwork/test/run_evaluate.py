import pickle
import os,re
import sys
import time
from decimal import Decimal
import glob
import subprocess
import numpy as np

if __name__=="__main__":

    dataDir = '/store/user/bfrancis/images_v5/SingleEle_2017F/'
    outDir = '/store/user/llavezzo/disappearingTracks/SingleEle2017F_validation/test/'
    logDir = '/data/users/llavezzo/Logs/evaluate/'
    reprocessAllFiles = True
    modelFile = '/share/scratch0/llavezzo/CMSSW_11_1_3/src/DisappTrksML/DeepSets/test/kfold_results/trainV2_param5_lessepochs/model.h5'

    if(not os.path.isdir(outDir)): os.mkdir(outDir)
    if(not os.path.isdir(logDir)): os.mkdir(logDir)

    alreadyProcessedFiles = []
    for filename in os.listdir(outDir):
        if('.npz' in filename and 'hist' in filename):
            index1 = filename.find("_")
            index2 = filename.find(".root")
            numFile = int(filename[index1+1:index2])
            alreadyProcessedFiles.append(numFile)
    files = []
    for filename in os.listdir(dataDir):
        if('.root' in filename and 'hist' in filename):
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
    request_disk = 500MB
    request_memory = 2GB
    request_cpus = 1
    executable              = eval_wrapper.sh
    arguments               = $(PROCESS) {1} {2} {3} {4}
    log                     = {5}log_$(PROCESS).log
    output                  = {5}out_$(PROCESS).txt
    error                   = {5}error_$(PROCESS).txt
    when_to_transfer_output = ON_EXIT
    getenv = true
    queue {0}
    """.format(len(files),filelist,modelFile,dataDir,outDir,logDir)

    f.write(submitLines)
    f.close()

    os.system('condor_submit run.sub')
