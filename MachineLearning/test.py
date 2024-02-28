import sys
import os
import pickle
import logging
sys.path.append("/data/users/mcarrigan/home_link/.local/bin/")
#sys.path.append('/home/ryan/Documents/Research/')
import optuna
from networkController import NetworkController

logging.basicConfig(level=logging.DEBUG)

params = [{"trainable_params":{"phi_layers" : ["layers", 1, 3, 8, 16]}},
          {"trainable_params":{"f_layers" : ["layers", 1, 3, 8, 16]}}]

with open("params.pkl", "wb") as pickle_file:
    pickle.dump(params, pickle_file)

NetworkController.generate_condor_submission(
    argument="$(PROCESS)",
    number_of_jobs=len(params),
    use_gpu=True, log_dir=".",
    input_files=["run_wrapper_gpu.sh", "runNN.py",
                 "ElectronModel.py", "networkController.py",
                 "generator.py", "singularity_wrapper.sh"
                 ])

os.system("condor_submit run.sub")
