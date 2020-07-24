import pickle
import os,re
import sys
import time
from decimal import Decimal
import glob
import subprocess
from itertools import product 

if __name__=="__main__":

    folder = "undersample_filters_256_512"

    f = open('run.sub', 'w')
    submitLines = """

    Universe = vanilla
    +IsLocalJob = true
    Rank = TARGET.IsLocalSlot
    request_disk = 250MB
    request_memory = 2GB
    request_cpus = 4
    executable              = run_wrapper.sh
    arguments               = {0}
    log                     = /data/users/llavezzo/Logs/cnn/log.log
    output                  = /data/users/llavezzo/Logs/cnn/out.txt
    error                   = /data/users/llavezzo/Logs/cnn/error.txt
    should_transfer_files   = Yes
    when_to_transfer_output = ON_EXIT
    transfer_input_files = {1}run_wrapper.sh, {1}flow.py, {1}utils.py, {1}validate.py
    getenv = true
    queue 1

    """.format(folder, "/share/scratch0/llavezzo/CMSSW_11_1_2_patch1/src/DisappTrksML/CNN/")

    f.write(submitLines)
    f.close()

    os.system('condor_submit run.sub')