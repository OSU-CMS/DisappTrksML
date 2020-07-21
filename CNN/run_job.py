import pickle
import os,re
import sys
import time
from decimal import Decimal
import glob
import subprocess
from itertools import product 

if __name__=="__main__":

    folder = "undersample""
    logTag = "_"+folder

    f = open('run.sub', 'w')
    submitLines = """

    Universe = vanilla
    +IsLocalJob = true
    Rank = TARGET.IsLocalSlot
    request_disk = 250
    request_memory = 2GB
    request_cpus = 4
    executable              = run_wrapper.sh
    arguments               = {0}
    log                     = /data/users/llavezzo/Logs/cnn{2}/log.log
    output                  = /data/users/llavezzo/Logs/cnn{2}/out.txt
    error                   = /data/users/llavezzo/Logs/cnn{2}/error.txt
    should_transfer_files   = Yes
    when_to_transfer_output = ON_EXIT
    transfer_input_files = {1}run_wrapper.sh, {1}flow.py, {1}utils.py, {1}validate.py
    getenv = true
    queue {0}

    """.format(folder, "/home/llavezzo/CMSSW_10_2_20/src/work/", logTag)

    f.write(submitLines)
    f.close()

    os.system('condor_submit run.sub')