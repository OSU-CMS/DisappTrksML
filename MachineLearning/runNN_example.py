"""
An example script of how you would use NetworkController to
tune hyprparameters
"""

import sys
sys.path.append("/data/users/mcarrigan/home_link/.local/")
import pickle

from networkController import NetworkController
from DisappTrksML.DeepSets.python.ElectronModel import ElectronModel

index = sys.argv[1]
print("Index: ", index)
controller = NetworkController(ElectronModel)

with open("params.pkl", "rb") as pickle_file:
    params = pickle.load(pickle_file)


data_directory="/home/ryan/Documents/Research/Data/DeepSetsTraining/TrainDataSample/"


controller.tune_hyperparameters(trainable_params= params[index],
                                train_parameters={"epochs" : 10,
                                                  "val_generator_params": {"input_dir": data_directory,
                                                                           "info_indices": [4,8,9,12],
                                                                           "batch_size": 256},
                                                  "train_generator_params": {"input_dir": data_directory,
                                                                             "info_indices":[4,8,9,12],
                                                                             "batch_size":256},
                                                  "use_tensorboard": True},
                                build_parameters={},
                                use_gpu=False,
                                input_dir=data_directory)
