import configparser

from networkController import NetworkController
from DisappTrksML.DeepSets.python.ElectronModel import ElectronModel

config = None

if not config:
    raise ValueError("Config should be defined if you are trying to use condor")

controller = NetworkController(ElectronModel(), config=config)
controller.tune_hyperparameters()
