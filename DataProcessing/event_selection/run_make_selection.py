import pickle
import os,re
import sys
import time
from decimal import Decimal
import glob
import subprocess
import numpy as np

if __name__=="__main__":

    workDir = ""

    dataDir = "/store/user/mcarrigan/disappearingTracks/converted_DYJetsToLL_M50_V3/"

    files = []
    for filename in os.listdir(dataDir):
        if('images_bkg_' in filename and '.npz' in filename):
            index1 = filename.find("0p25_")
            index2 = filename.find(".")
            numFile = int(filename[index1+5:index2])
            files.append(numFile)
    np.save('fileslist',files)
    

    f = open('run.sub', 'w')
    submitLines = """

    Universe = vanilla
    +IsLocalJob = true
    Rank = TARGET.IsLocalSlot
    request_disk = 500MB
    request_memory = 2048MB
    request_cpus = 1
    executable              = selection_wrapper.sh
    arguments               = $(PROCESS)
    log                     = /data/users/mcarrigan/Logs/event_selection/log_$(PROCESS).log
    output                  = /data/users/mcarrigan/Logs/event_selection/out_$(PROCESS).txt
    error                   = /data/users/mcarrigan/Logs/event_selection/error_$(PROCESS).txt
    should_transfer_files   = Yes
    when_to_transfer_output = ON_EXIT
    transfer_input_files = {0}fileslist.npy, {0}selection_wrapper.sh, {0}make_selectionME.py
    getenv = true
    queue {1}

    """.format(workDir,len(files))

    f.write(submitLines)
    f.close()

    os.system('condor_submit run.sub')
