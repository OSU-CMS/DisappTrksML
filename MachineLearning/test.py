import sys
import pickle
import logging


sys.path.append('/home/ryan/Documents/Research/')
from networkController import NetworkController

logging.basicConfig(level=logging.DEBUG)

params = [{"trainable_params":{"phi_layers" : ["layers", 1, 3, 8, 16]}},
          {"trainable_params":{"f_layers" : ["layers", 1, 3, 8, 16]}}]

with open("params.pkl", "wb") as pickle_file:
    pickle.dump(params, pickle_file)

NetworkController.generate_condor_submission(
    argument="'python3 runNN.py $(PROCESS)'",
    number_of_jobs=len(params),
    use_gpu=True, log_dir=".",
    input_files=["run_wrapper_gpu.sh", "runNN.py",
                 "ElectronModel.py", "networkController.py",
                 "generator.py"
                 ])

