import sys
sys.path.append('/home/ryan/Documents/Research/DisappearingTracks/')
from networkController import NetworkController
from DisappTrksML.DeepSets.python.ElectronModel import ElectronModel
import logging
logging.basicConfig(level=logging.DEBUG)

data_directory="/home/ryan/Documents/Research/Data/DeepSetsTraining/TrainDataSample/"


controller = NetworkController(ElectronModel())
# controller.build_model
controller.tune_hyperparameters(trainable_params={"phi_layers": ["layers", 1, 3, 8, 16]},
                                train_parameters={"epochs" : 1,
                                                  "val_generator_params": {"input_dir": data_directory,
                                                                           "info_indices": [4,8,9,12],
                                                                           "batch_size": 256},
                                                  "train_generator_params": {"input_dir": data_directory,
                                                                             "info_indices":[4,8,9,12],
                                                                             "batch_size":256}},
                                build_parameters={},
                                use_gpu=False,
                                input_dir=data_directory)

