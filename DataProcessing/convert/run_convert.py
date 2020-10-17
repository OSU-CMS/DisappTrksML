import pickle
import os,re
import sys
import time
from decimal import Decimal
import glob
import subprocess
import numpy as np

if __name__=="__main__":

    dataDir = '/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v4_selection_muons/'
    outDir = '/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v4_sets_muons_MUO/'
    logDir = '/data/users/llavezzo/Logs/convert/'
    reprocessAllFiles = True

    if(not os.path.isdir(outDir)): os.mkdir(outDir)

    if(os.path.exists(dataDir+'sCounts.pkl')):
        os.system("cp "+dataDir+"sCounts.pkl "+outDir)
    else: print("Missing sCounts.pkl file!")
    if(os.path.exists(dataDir+'bkgCounts.pkl')):
        os.system("cp "+dataDir+"bkgCounts.pkl "+outDir)
    else: print("Missing bkgCounts.pkl file!")

    alreadyProcessedFiles = []
    for filename in os.listdir(outDir):
        if('.root' in filename and 'hist' in filename):
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
    arguments               = $(PROCESS) {1} {2} {3}
    log                     = {4}log_$(PROCESS).log
    output                  = {4}out_$(PROCESS).txt
    error                   = {4}error_$(PROCESS).txt
    should_transfer_files   = Yes
    when_to_transfer_output = ON_EXIT
    transfer_input_files = {2}, wrapper.sh, toSets.py
    getenv = true
    queue {0}
    """.format(len(files),dataDir,filelist,outDir, logDir)

    f.write(submitLines)
    f.close()

    os.system('condor_submit run.sub')
