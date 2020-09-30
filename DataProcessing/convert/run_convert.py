import pickle
import os,re
import sys
import time
from decimal import Decimal
import glob
import subprocess
import numpy as np

if __name__=="__main__":

    dataDir = '/store/user/mcarrigan/disappearingTracks/images_DYJetsToLL_M50/'
    outDir = ''
    files = []
    for filename in os.listdir(dataDir):
        if('.root' in filename and 'hist' in filename):
            index1 = filename.find("_")
            index2 = filename.find(".")
            numFile = int(filename[index1+1:index2])
            files.append(numFile)
            break
    filelist = 'fileslist.txt'
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
    log                     = /data/users/llavezzo/Logs/convert/log_$(PROCESS).log
    output                  = /data/users/llavezzo/Logs/convert/out_$(PROCESS).txt
    error                   = /data/users/llavezzo/Logs/convert/error_$(PROCESS).txt
    should_transfer_files   = Yes
    when_to_transfer_output = ON_EXIT
    transfer_input_files = {1}, wrapper.sh, toSets.py
    getenv = true
    queue {0}

    """.format(len(files),file,dataDir,filelist,outDir)

    f.write(submitLines)
    f.close()

    os.system('condor_submit run.sub')