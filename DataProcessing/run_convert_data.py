import pickle
import os,re
import sys
import time
from decimal import Decimal
import glob
import subprocess

def submitOneJob(index, lower_i, upper_i):
    
    f = open('run.sub', 'w')
    submitLines = """

    Universe = vanilla
    +IsLocalJob = true
    Rank = TARGET.IsLocalSlot
    request_disk = 2MB
    request_memory = 2048MB
    request_cpus = 1
    executable              = run_wrapper.sh
    arguments               = {0} {1} {2}
    log                     = /data/users/llavezzo/Logs/convert_data/log_{0}_$(PROCESS).log
    output                  = /data/users/llavezzo/Logs/convert_data/out_{0}_$(PROCESS).txt
    error                   = /data/users/llavezzo/Logs/convert_data/error_{0}_$(PROCESS).txt
    should_transfer_files   = Yes
    when_to_transfer_output = ON_EXIT
    transfer_input_files = /home/llavezzo/DisappTrksML/DataProcessing/run_wrapper.sh, /home/llavezzo/DisappTrksML/DataProcessing/convert_data.py, /home/llavezzo/DisappTrksML/TreeMaker/interface/Infos.h
    getenv = true
    queue 1

    """.format(index, lower_i, upper_i)

    f.write(submitLines)
    f.close()

    #os.system('touch fileName.txt')
    os.system('condor_submit run.sub')

if __name__=="__main__":

    batches = 55
    entries = 550000

    events_per_batch = int(1.0*entries/batches)
    
    for i in range(batches):
        print("Submitting events",i*events_per_batch,"-",(i+1)*events_per_batch)
        submitOneJob(i, i*events_per_batch,(i+1)*events_per_batch)
