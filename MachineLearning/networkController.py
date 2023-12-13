#!/usr/bin/env python3
""" This module is used to control a general neural network """
from abc import ABC, abstractmethod
import os
from typing import Union, Dict
import numpy as np
import cmsml
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import optuna
import logging
import keras
from glob import glob


class NetworkBase(ABC):
    """
    This abstract class requires that the following methods be defined in your network:
    build_model() - responsible for creating layers, nodes, and activation functions in model
    evaluate_model() - Used to run a single data file through the trained model
    train_model() - Used to train the model on a directory of data

    Your model should also inherit this class
    """
    def __init__(self):
        print("awhlkdsajflksajdf")
    @abstractmethod
    def build_model(self, *args, **kwargs):
        """ Return a built nueral network """
        pass

    @abstractmethod
    def evaluate_model(self, fileName:str, *args, **kwargs)->Union[list[float],None]:
        """ Return prediction of neural network on input data """
        pass

    @abstractmethod
    def train_model(self, model:Model, data_directory:str, epochs:int = 10, monitor:str='val_loss', patience_count:int=10, metrics:list=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()], outdir:str="", *args, **kwargs)->Model:
        """ Return trained neural network """
        pass

    @abstractmethod
    def get_metrics(self, input_directory:str, threshold:float = 0.5, glob_patterns="*", **kwargs)->list[int]:
        """ Return list of TP, TN, FP, FN"""
        pass

    def evaluate_directory(self, directory_path:str, glob_pattern:Union[None, str]=None, **kwargs)->Union[np.ndarray, None]:
        """
        Conveineince function to evaluate the model on files in a directory using a specific glob pattern
        """
        if glob_pattern:
            inputFiles = glob(directory_path+glob_pattern)
        else:
            inputFiles = glob(directory_path+"*")

        results = np.empty(1) # To avoid unbound variable

        for i, file in enumerate(inputFiles):
            if i % 100 == 0:
                logging.info(f"On file {i}")
                eval_output = self.evaluate_model(file, **kwargs)
                print(eval_output)
                if eval_output is None:
                    return None
                if i == 0:
                    results = np.array(eval_output)
                else:
                    results = np.concatenate((results, np.array(eval_output)))
        return results
                    

class NetworkController():
    def __init__(self, model:NetworkBase, config:Union[str,None]=None):

        self.model = model
        # if not issubclass(type(self.model), NetworkBase):
        #     print(issubclass(type(self.model), NetworkBase))
        #     print(type(model))
        #     raise TypeError("You model needs to subclass the NetworkBase class!")
        # self.config = config
        self.input_dir = None

        
    def calculateMetrics(self, metrics:list[int])->tuple[float, float, float]:
        """
        Return precision, recall, and f1 score

        The input should be a list of the following order [TP, TN, FP, FN]
        """
        if metrics[0] == 0: 
            logging.warning("No TP found!")
            return 0, 0, 0
        else:
            precision = metrics[0] / (metrics[0] + metrics[2])
            recall = metrics[0] / (metrics[0] + metrics[3])
            f1 = 2 * (precision * recall) / (precision + recall)
            return precision, recall, f1

    def save_model(self, model:Model):
        cmsml.tensorflow.save_graph("graph.pb", model, variables_to_constants=True)
        cmsml.tensorflow.save_graph("graph.pb.txt", model, variables_to_constants=True)

    def load_model(self, model_path:str)->Model:
        """ Return a pretrained model that was previously stored in a h5 file """
        return load_model(model_path)

    def _objective(self, trial, trainable_params, train_parameters, build_parameters, glob_pattern, metric, threshold = 0.5):
        """
        This method should not be used externally. This function is used by optuna
        to optimize the model.
        """

        if not self.input_dir:
            logging.debug("""You should not be calling this function outside of tune_hyprparameters.
                           The input_directory for NetworkController needs to be set""")
            return
        output_values = {} 
        for key, value in trainable_params.items():
            if value[0] == "int":
                output_values[key] = trial.suggest_int(str(key), value[1], value[2])
            elif value[0] == "float":
                output_values[key] = trial.suggest_float(str(key), value[1], value[2])
            elif value[0] == "category":
                output_values[key] = trial.suggest_categorical(str(key), value)
            elif value[0] == "layers":
                number_of_layers = trial.suggest_int("num_layers", value[1], value[2])
                output_values[key] = [trial.suggest_int(f"nodes_{i}", value[3], value[4], log=True) for i in
                                      range(number_of_layers)]

        self.model.build_model(**build_parameters, **output_values)
        logging.debug(f"train_model parameters {train_parameters}")
        self.model.train_model(self.input_dir, **train_parameters)
        return self.calculateMetrics(self.model.get_metrics(self.input_dir, threshold, glob_pattern))[metric]
        
        

    def tune_hyperparameters(self,
                             trainable_params: Union[Dict[str, tuple[float,float]],
                                               Dict[str, tuple[int, int]], Dict[str, list]] = {},
                             train_parameters = {},
                             build_parameters = {},
                             use_gpu:bool=True, use_condor:bool=False, num_trials:int= 10, timeout:int=600,
                             input_dir:str="", glob_pattern="*", metric = 2, threshold = 0.5)->None:
        """
        Return output of hyperparameter tuning of your model. Type checking is performed in the objective function, so make sure
        that you properly distinguish floats from integers when inputting the trainable params

        """
        if use_gpu:
            config = tf.compat.v1.ConfigProto(log_device_placement=True)
            tf.compat.v1.Session(config=config)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            # Limits CPU
            config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads = 4,
                                              intra_op_parallelism_threads = 4,
                                              allow_soft_placement = True,
                                              device_count={'CPU':4})
            tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        self.input_dir = input_dir 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        logging.debug("Setting up Optuna study")
        study = optuna.create_study(direction="maximize")

        study.optimize(lambda trials: self._objective(trials, train_parameters=train_parameters,
                                                      build_parameters=build_parameters,
                                                      trainable_params = trainable_params,
                                                      glob_pattern=glob_pattern, metric=metric, threshold=threshold),
                       num_trials, timeout) 
        
