#!/usr/bin/env python3
""" This module is used to control a general neural network """
from abc import ABC, abstractmethod
from typing import Union
import cmsml
from tensorflow.keras.models import Model, load_model
import logging
import keras

class NetworkController(ABC):
    """
    This abstract class requires that the following methods be defined in your network:
    buildModel() - responsible for creating layers, nodes, and activation functions in model


    Your model should also inherit this class
    """

    def calculateMetrics(metrics:list[int, int, int, int])->tuple[float, float, float]:
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

    def saveModel(self):
        cmsml.tensorflow.save_graph("graph.pb", self.model, variables_to_constants=True)
        cmsml.tensorflow.save_graph("graph.pb.txt", self.model, variables_to_constants=True)

    def load_model(self, model_path:str)->Model:
        """ Return a pretrained model that was previously stored in a h5 file """
        return load_model(model_path)
    
    @abstractmethod
    def buildModel(self, *args, **kwargs):
        """ Return a built nueral network """
        logging.warning("buildModel() should be implemented in your class!")

    @abstractmethod
    def evaluateModel(self, fileName:str, *args, **kwargs)->Union[list[float],None]:
        """ Return prediction of neural network on input data """
        logging.warning("evaluateModel() should be implemented in your class!")

    @abstractmethod
    def trainModel(self, model:Model, data_directory:str, epochs:int = 10, monitor:str='val_loss', patience_count:int=10, metrics:list=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()], outdir:str="", *args, **kwargs)->Model:
        """ Return trained neural network """
        logging.warning("evaluateModel() should be implemented in your class!")

