import os
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
import json
import random
import sys
import pickle
import datetime
import getopt
import plotMetrics
from datetime import date
import utilities
from fakeClass import fakeNN
from ROOT import TTree, TBranch, TFile
import argparse

class Validator:

    workDir = os.getcwd()
    outputDir = workDir + '/outputFiles/'
    dataDir = ['/store/user/mcarrigan/fakeTracks/converted_DYJets-MC2022_v1/']
    weightsDir = ''

    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.initialize()
        self.configInfo()

    def initialize(self):

        if(self.args.outputDir):
            Validator.outputDir = self.args.outputDir
        if(self.args.dataDir):
            Validator.dataDir = [self.args.dataDir]
        if(self.args.weightsDir):
            Validator.weightsDir = self.args.weightsDir
        else:
            Validator.weightsDir = self.config['weightsDir']

        # create output directory
        if not os.path.exists(Validator.outputDir): os.system('mkdir '+str(Validator.outputDir))

    def configInfo(self):
        print(utilities.bcolors.BLUE + "Using the following config options:")
        print('\t Weights Directory: \n \t \t{}'.format(self.config['weightsDir']))
        print('\t Data Directory: {}'.format(self.config['dataDir']))
        print('\t Plot Name: {}'.format(self.config['plotsName']))
        print('\t Validation Metrics: {}'.format(self.config['val_metrics']))
        print('\t Batch Normalization: {}'.format(self.config['batch_norm']))
        print('\t Epochs: {}'.format(self.config['epochs']))
        print('\t Filters: {}'.format(self.config['filters']))
        print('\t Input Dimension: {}'.format(self.config['input_dim']))
        print('\t Undersampling: {}'.format(self.config['undersample']))
        print('\t Dropout: {}'.format(self.config['dropout']))
        print('\t Deleted Elements: {}'.format(self.config['delete_elements']))
        print('\t Categories Used: {}'.format(self.config['saveCategories']))
        print('\t Data Normalization: {}'.format(self.config['normalize_data']))
        print('\t Threshold: {}'.format(self.config['threshold']))
        print('\t Debugging: {}'.format(self.config['DEBUG']) + utilities.bcolors.ENDC)

    def loadModel(self, name='lastEpoch.h5'):
        weights = Validator.weightsDir
        if not Validator.weightsDir.endswith('h5'):
            weights += name
        self.model = keras.models.load_model(weights)    

    def gatherData(self):
        self.inputs, self.input_dim = utilities.getInputs(self.config['input_dim'], self.config['delete_elements'])

        for i, dataSet in enumerate(Validator.dataDir):
            if i == 0:
                #swapped test/train here because we want to validate on "test" data
                testTracks, trainTracks, valTracks, testTruth, trainTruth, valTruth = utilities.loadData(
                    str(dataSet), self.config['undersample'], self.inputs, self.config['normalize_data'], 
                    self.config['saveCategories'][i], 0, 0, self.args.test)
            else:
                testTracks2, trainTracks2, valTracks2, testTruth2, trainTruth2, valTruth2 = utilities.loadData(
                    str(dataSet), self.config['undersample'], self.inputs, self.config['normalize_data'], 
                    self.config['saveCategories'][i], 0, 0, self.args.test) 
                trainTracks = np.concatenate((trainTracks, trainTracks2))
                trainTruth = np.concatenate((trainTruth, trainTruth2))
                testTracks = np.concatenate((testTracks, testTracks2))
                testTruth = np.concatenate((testTruth, testTruth2))
                valTracks = np.concatenate((valTracks, valTracks2))
                valTruth = np.concatenate((valTruth, valTruth2))

        indices = np.arange(len(testTracks))
        np.random.shuffle(indices)   
        testTracks = testTracks[indices]
        self.testTruth = testTruth[indices]

        self.eventNumbers = testTracks[:, 0]
        self.testTracks = testTracks[:, 1:]

    def passData(self, tracks, events, truth):
        self.testTracks = tracks
        self.eventNumbers = events
        self.testTruth = truth

    def makePredictions(self, inputs=None, save=True):
        if inputs is None:
            inputs = self.inputs
        self.predictions = self.model.predict(self.testTracks)
        pred_fakes = np.argwhere(self.predictions >= self.config['threshold'])
        pred_reals = np.argwhere(self.predictions < self.config['threshold'])
        self.predictionsBinary = [1 if x >= self.config['threshold'] else 0 for x in self.predictions]
        self.input_variables = utilities.listVariables(inputs)

        print(utilities.bcolors.BLUE + "Number of predicted fakes: " + str(len(pred_fakes)) + ", Number of predicted Reals: " + str(len(pred_reals)) + utilities.bcolors.ENDC)
        if save:
            np.savez_compressed(Validator.outputDir + "predictions.npz", tracks = self.testTracks, truth = self.testTruth, 
                                predictions = self.predictions, inputs = inputs, events = self.eventNumbers)
        if os.path.exists(Validator.outputDir + 'validateOutput.root'): 
            os.remove(Validator.outputDir + 'validateOutput.root')
        self.classifications = plotMetrics.getStats(self.testTruth, self.predictions, Validator.outputDir, outputfile='validateOutput.root')
        return self.classifications

    def makePlots(self):

        #if os.path.exists(Validator.outputDir + 'validateOutput.root'): 
        #    os.remove(Validator.outputDir + 'validateOutput.root')
        plotMetrics.plotCM(self.testTruth, self.predictionsBinary, Validator.outputDir, outputfile='validateOutput.root')
        plotMetrics.plotScores(self.predictions, self.testTruth, self.config['plotsName'], Validator.outputDir, outputfile='validateOutput.root')
        #self.classifications = plotMetrics.getStats(self.testTruth, self.predictions, Validator.outputDir, plot=True, outputfile='validateOutput.root')

        d0Index = np.where(self.input_variables.astype(str) == 'd0')
        plotMetrics.backgroundEstimation(self.testTracks, self.predictions, d0Index, Validator.outputDir, outputfile='validateOutput.root')
        plotMetrics.makeSkim(self.testTracks, self.predictions, self.testTruth, self.eventNumbers, Validator.outputDir, outputfile='validateOutput.root')


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('-o', '--outputDir', type=str, default='', help='Output directory to save output files and plots')
    parser.add_argument('-d', '--dataDir', type=str, default='', help='Input data directory')
    parser.add_argument('-c', '--config', type=str, default='', help='Config file of inputs for network (json)')
    parser.add_argument('-w', '--weightsDir', type=str, default='', help='Weights directory containing model weights')
    parser.add_argument('-t', '--test', action='store_true', help='Run in debug mode')

    args = parser.parse_args()
    return args



if __name__ == "__main__":

    args = parse_args()

    ################default config parameters################
    weightsDir = '/data/users/mcarrigan/fakeTracks/networks/dropoutSearch/fakeTracks_4PlusLayer_aMCv9p3_NGBoost_ProducerValidation_3-29_v2/fakeTracks_4PlusLayer_aMCv9p3_NGBoost_ProducerValidation_3-29_v2_p7/weights/'
    plotsName = 'validation_FakeProducer'
    workDir = '/data/users/mcarrigan/fakeTracks/'
    dataDir = ['/store/user/mcarrigan/fakeTracks/converted_DYJets-MC2022_v1/']
    val_metrics = [keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
    batch_norm = False
    batch_size = 64
    epochs = 1
    filters = [16, 8]
    input_dim = 178
    undersample = -1
    dropout = 0.1
    delete_elements = ['passesSelection']
    saveCategories = [{'fake':True, 'real':True, 'pileup':False}]
    normalize_data = False
    threshold = 0.5
    DEBUG = True
    #################################################

    config_dict = {}

    if args.config: config_dict = utilities.readConfig(configFile)
    else:
        config_dict = {'weightsDir': weightsDir, 
                    'plotsName' : plotsName,
                    'dataDir' : dataDir,
                    'val_metrics' : val_metrics,
                    'batch_norm' : batch_norm,
                    'epochs' : epochs,
                    'filters' : filters,
                    'input_dim' : input_dim,
                    'undersample' : undersample,
                    'dropout' : dropout,
                    'delete_elements' : delete_elements,
                    'saveCategories' : saveCategories,
                    'normalize_data' : normalize_data,
                    'threshold' : threshold,
                    'DEBUG' : DEBUG}

    myValidator = Validator(config_dict, args)

    myValidator.gatherData()
    myValidator.loadModel()
    myValidator.makePredictions()
    myValidator.makePlots()





