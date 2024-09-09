"""
An example script of how you would use NetworkController to
tune hyprparameters
"""

import sys
import os
import pickle

import numpy

from networkController import NetworkController
from ElectronModel import ElectronModel

index = int(sys.argv[1])
print("Index: ", index)
model = ElectronModel() # Need this so can access class properties correctly

controller = NetworkController(model)

params = numpy.load("training_params.npy", allow_pickle=True)

# If running in the singularity container, bind the directory you want to use for data
# input to /data in the container.
training_directory="/data/training/"
testing_directory="/data/testing/"
logging_directory="/store/"

controller.tune_hyperparameters(trainable_params= params[index],
                                train_parameters={"epochs" : 1,
                                                  "val_generator_params": {"input_dir": training_directory,
                                                                           "info_indices": [4,8,9,12],
                                                                           "batch_size": 256},
                                                  "train_generator_params": {"input_dir": training_directory,
                                                                             "info_indices":[4,8,9,12],
                                                                             "max_hits" : 100,
                                                                             "batch_size":256},
                                                  "use_tensorboard": True},
                                build_parameters={},
                                use_gpu=True,
                                num_trials = 1,
                                log_dir = logging_directory,
                                input_dir=training_directory,
                                testing_dir=testing_directory)
