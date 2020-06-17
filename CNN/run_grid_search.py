import pickle
import os,re
import sys
import time
from decimal import Decimal
import glob
import subprocess
from itertools import product 

if __name__=="__main__":

    filters_list = [32, 64, 128]  
    layers_list = [2,5,10]
    optimizer = ['adam','adadelta']
    parameters = list(product(filters_list, layers_list, optimizer))
    njobs = len(parameters)

    f = open('run.sub', 'w')
    submitLines = """

    Universe = vanilla
    +IsLocalJob = true
    Rank = TARGET.IsLocalSlot
    request_disk = 2MB
    request_memory = 2048MB
    request_cpus = 1
    executable              = run_kfold.sh
    arguments               = $(PROCESS)
    log                     = /data/users/llavezzo/Logs/cnn_gs/log_$(PROCESS).log
    output                  = /data/users/llavezzo/Logs/cnn_gs/out_$(PROCESS).txt
    error                   = /data/users/llavezzo/Logs/cnn_gs/error_$(PROCESS).txt
    should_transfer_files   = Yes
    when_to_transfer_output = ON_EXIT
    transfer_input_files = /home/llavezzo/DisappTrksML/CNN/run_kfold.sh, /home/llavezzo/DisappTrksML/CNN/kfold.py, /home/llavezzo/DisappTrksML/CNN/utils.py
    getenv = true
    queue {0}

    """.format(njobs)

    f.write(submitLines)
    f.close()

    os.system('condor_submit run.sub')