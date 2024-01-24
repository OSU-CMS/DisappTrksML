import sys
import pickle
import logging


sys.path.append('/home/ryan/Documents/Research/')
from networkController import NetworkController

logging.basicConfig(level=logging.DEBUG)

params = [{"trainable_params":{"phi_layers" : ["layers", 1, 3, 8, 16]}},
          {"trainable_params":{"f_layers" : ["layers", 1, 3, 8, 16]}}]

controller = NetworkController(ElectronModel())
# controller.build_model
controller.tune_hyperparameters(trainable_params={"phi_layers": ["layers", 1, 3, 8, 16]},
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

