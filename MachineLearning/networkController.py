#!/usr/bin/env python3
""" This module is used to control a general neural network """
from abc import ABC, abstractmethod
import typing
import logging

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

       
    
    @abstractmethod
    def buildModel(**kwargs):
        """ Build neural network """
        logging.warning("buildModel() should be implemented in your class!")

    @abstractmethod
    def evaluateModel():
        """ Return prediction of neural network on input data """
        logging.warning("evaluateModel() should be implemented in your class!")

    @abstractmethod
    def trainModel(**kwargs):
        """ Train neural network """
        logging.warning("evaluateModel() should be implemented in your class!")
