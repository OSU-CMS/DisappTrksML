#!/usr/bin/env python3
""" This module is used to control a general neural network """
from abc import ABC, abstractmethod
import os
from typing import Union
import cmsml
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import optuna
import logging
import keras

class NetworkBase(ABC):
    """
    This abstract class requires that the following methods be defined in your network:
    build_model() - responsible for creating layers, nodes, and activation functions in model
    evaluate_model() - Used to run a single data file through the trained model
    train_model() - Used to train the model on a directory of data

    Your model should also inherit this class
    """

    @abstractmethod
    def build_model(self, *args, **kwargs):
        """ Return a built nueral network """
        logging.warning("buildModel() should be implemented in your class!")

    @abstractmethod
    def evaluate_model(self, fileName:str, *args, **kwargs)->Union[list[float],None]:
        """ Return prediction of neural network on input data """
        logging.warning("evaluateModel() should be implemented in your class!")

    @abstractmethod
    def train_model(self, model:Model, data_directory:str, epochs:int = 10, monitor:str='val_loss', patience_count:int=10, metrics:list=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()], outdir:str="", *args, **kwargs)->Model:
        """ Return trained neural network """
        logging.warning("evaluateModel() should be implemented in your class!")


class NetworkController():
    def __init__(self, model:NetworkBase):
        if not isinstance(model, NetworkBase):
            raise TypeError("You model needs to subclass the NetworkBase class!")
        self.model = model

        
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

    def tune_hyperparameters(self, model:Model, use_gpu=True, use_condor=False)->Model:
        
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
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        study = optuna.create_study(direction="maximize")
        study.optimize(
        
        return 
