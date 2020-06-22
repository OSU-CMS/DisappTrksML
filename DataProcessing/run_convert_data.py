import pickle
import os,re
import sys
import time
from decimal import Decimal
import glob
import subprocess

if __name__=="__main__":

    f = open('run.sub', 'w')
    submitLines = """

    Universe = vanilla
    +IsLocalJob = true
    Rank = TARGET.IsLocalSlot
    request_disk = 500MB
    request_memory = 2048MB
    request_cpus = 1
    executable              = run_wrapper.sh
    arguments               = $(PROCESS)
    log                     = /data/users/llavezzo/Logs/convert_data/log_$(PROCESS).log
    output                  = /data/users/llavezzo/Logs/convert_data/out_$(PROCESS).txt
    error                   = /data/users/llavezzo/Logs/convert_data/error_$(PROCESS).txt
    should_transfer_files   = Yes
    when_to_transfer_output = ON_EXIT
    transfer_input_files = {0}DataProcessing/run_wrapper.sh, {0}DataProcessing/convert_data.py, {0}TreeMaker/interface/Infos.h
    getenv = true
    queue 4000

    """.format("/home/llavezzo/CMSSW_10_2_20/src/DisappTrksML/")

    f.write(submitLines)
    f.close()

    os.system('condor_submit run.sub')
