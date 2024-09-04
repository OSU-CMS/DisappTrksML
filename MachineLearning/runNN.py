"""
An example script of how you would use NetworkController to
tune hyprparameters
"""

import sys
import pickle

import numpy

from networkController import NetworkController
from DisappTrksML.DeepSets.python.ElectronModel import ElectronModel

index = int(sys.argv[1])
print("Index: ", index)
controller = NetworkController(ElectronModel)

params = numpy.load("training_params.npy", allow_pickle=True)


data_directory="/data"


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
