import os
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from keras import backend as K
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import json
import random
import sys
import pickle
import datetime
import getopt
import plotMetrics
from datetime import date
from sklearn.preprocessing import MinMaxScaler
import utilities
from fakeClass import fakeNN
import cmsml
import argparse
from validateData import Validator

class networkController:

    outputDir = 'outfakesNN_' + date.today().strftime('%m_%d')
    paramsFile = ""
    params = []
    paramsIndex = 0
    gridSearch = -1

    def __init__(self, config, args):
        self.config = config
        self.args = args

        self.initialize()

    def initialize(self):
        #self.cpu_settings()
        self.gpu_settings()

        if self.args.outputDir: networkController.outputDir = self.args.outputDir
        if self.args.paramsFile: networkController.paramsFile = self.args.paramsFile
        if self.args.index: networkController.paramsIndex = self.args.index
        if self.args.grid: networkController.gridSearch = self.args.grid

        if(len(networkController.paramsFile)>0):
            try:
                networkController.params = np.load(str(networkController.paramsFile), allow_pickle=True)[networkController.paramsIndex]
            except:
                print(utilities.bcolors.RED+"ERROR: Index outside range or no parameter list passed"+utilities.bcolors.ENDC)
                print(utilities.bcolors.RED+"USAGE: fakesNN.py -d/--dir= output_directory -p/--params= parameters.npy -i/--index= parameter_index"+utilities.bcolors.ENDC)
                sys.exit(2)
            if(networkController.gridSearch > 0):
                networkController.outputDir = networkController.outputDir + "_g" + str(networkController.gridSearch) + "_p" + str(networkController.paramsIndex)
            else:
                networkController.outputDir = networkController.outputDir + "_p" + str(networkController.paramsIndex)
        cnt=0
        while(os.path.isdir(networkController.outputDir)):
            cnt+=1
            if(cnt==1): networkController.outputDir = networkController.outputDir+"_"+str(cnt)
            else: networkController.outputDir = networkController.outputDir[:-1] + str(cnt)
        print(utilities.bcolors.YELLOW+"Output directory: "+networkController.outputDir+utilities.bcolors.ENDC)
        if(len(networkController.params) > 0): 
            print(utilities.bcolors.YELLOW+"Using params"+utilities.bcolors.ENDC, networkController.params, ' ')
            print(utilities.bcolors.YELLOW+"from file "+utilities.bcolors.ENDC)
        
        self.plotDir = networkController.outputDir + '/plots/'
        self.weightsDir = networkController.outputDir + '/weights/'
        self.filesDir = networkController.outputDir + '/outputFiles/'

        if not os.path.exists(networkController.outputDir):
            os.mkdir(networkController.outputDir)
        if not os.path.exists(self.plotDir):
            os.mkdir(self.plotDir)
        if not os.path.exists(self.weightsDir):
            os.mkdir(self.weightsDir)
        if not os.path.exists(self.filesDir):
            os.mkdir(self.filesDir)

    def cpu_settings(self):
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
        config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads = 4,   
                                        intra_op_parallelism_threads = 4,
                                        allow_soft_placement = True,
                                        device_count={'CPU': 4})
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

        # suppress warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    def gpu_settings(self):
        config=tf.compat.v1.ConfigProto(log_device_placement=True)
        sess = tf.compat.v1.Session(config=config)
        print(sess.run(c))

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

    def getInputs(self):
        self.inputs, self.input_dim = utilities.getInputs(self.config['input_dim'], self.config['delete_elements'])

    def loadData(self):
        for i, dataSet in enumerate(self.config['dataDir']):
            if i == 0:
                if(self.config['loadSplitDataset']):
                    trainTracks, testTracks, valTracks, trainTruth, testTruth, valTruth = utilities.loadSplitDataset(str(dataSet), self.inputs)
                else:
                    trainTracks, testTracks, valTracks, trainTruth, testTruth, valTruth = utilities.loadData(str(dataSet), 
                                                                                                            self.config['undersample'], 
                                                                                                            self.inputs, 
                                                                                                            self.config['normalize_data'], 
                                                                                                            self.config['saveCategories'][i], 
                                                                                                            self.config['trainPCT'], 
                                                                                                            self.config['valPCT'], 
                                                                                                            self.args.test)
            else:
                if(self.config['loadSplitDataset']):
                    trainTracks2, testTracks2, valTracks2, trainTruth2, testTruth2, valTruth2 = utilities.loadSplitDataset(str(dataSet), self.inputs)
                else:
                    trainTracks2, testTracks2, valTracks2, trainTruth2, testTruth2, valTruth2 = utilities.loadData(str(dataSet), 
                                                                                                                self.config['undersample'], 
                                                                                                                self.inputs,
                                                                                                                self.config['normalize_data'], 
                                                                                                                self.config['saveCategories'][i], 
                                                                                                                self.config['trainPCT'], 
                                                                                                                self.config['valPCT'], 
                                                                                                                self.args.test)
                trainTracks = np.concatenate((trainTracks, trainTracks2))
                trainTruth = np.concatenate((trainTruth, trainTruth2))
                testTracks = np.concatenate((testTracks, testTracks2))
                testTruth = np.concatenate((testTruth, testTruth2))
                valTracks = np.concatenate((valTracks, valTracks2))
                valTruth = np.concatenate((valTruth, valTruth2))
        
        self.trainTracks = trainTracks
        self.trainTruth = trainTruth
        self.testTracks = testTracks
        self.testTruth = testTruth
        self.valTracks = valTracks
        self.valTruth = valTruth

        print("Train Tracks " + str(trainTracks.shape))
        print("Train Truth " + str(trainTruth.shape))
        print("Test Tracks " + str(testTracks.shape))
        print("Test Truth " + str(testTruth.shape))
        print("Val Tracks " + str(valTracks.shape))
        print("Val Truth " + str(valTruth.shape))

    def randomizeData(self):
        indices = np.arange(len(self.trainTracks))
        np.random.shuffle(indices)   
        trainTracks = self.trainTracks[indices]
        self.trainEvents = trainTracks[:, 0] #make array of only eventNumber
        self.trainTracks = trainTracks[:, 1:] #removing eventNumber from array
        self.trainTruth = self.trainTruth[indices]

        indices = np.arange(len(self.testTracks))
        np.random.shuffle(indices)
        testTracks = self.testTracks[indices]
        self.testEvents = trainTracks[:, 0]
        self.testTracks = testTracks[:, 1:]
        self.testTruth = self.testTruth[indices]

        indices = np.arange(len(self.valTracks))
        np.random.shuffle(indices)
        valTracks = self.valTracks[indices]
        self.valEvents = valTracks[:, 0]
        self.valTracks = valTracks[:, 1:]
        self.valTruth = self.valTruth[indices] 

    def setCallbacks(self):
        self.callbacks = [keras.callbacks.EarlyStopping(patience=self.config['patience_count']), 
                        keras.callbacks.ModelCheckpoint(filepath=self.weightsDir+'model.{epoch}.h5', 
                            save_best_only=self.config['save_best_only'], 
                            monitor=self.config['monitor'], 
                            mode=self.config['mode'])]


    def createModel(self):
        self.model = fakeNN(self.config['filters'],
                            self.input_dim,
                            self.config['batch_norm'],
                            self.config['val_metrics'],
                            self.config['dropout'])

        self.estimator = KerasClassifier(build_fn=self.model, 
                                        epochs=self.config['epochs'], 
                                        batch_size=self.config['batch_size'],
                                        verbose=1)

    def fitModel(self):
        self.history = self.estimator.fit(self.trainTracks, 
                                        self.trainTruth,
                                        validation_data=(self.valTracks, self.valTruth),
                                        callbacks=self.callbacks)

    def saveModel(self):
        cmsml.tensorflow.save_graph(self.filesDir+"graph.pb", self.estimator.model, variables_to_constants=True)
        cmsml.tensorflow.save_graph(self.filesDir+'graph.tb.txt', self.estimator.model, variables_to_constants=True)
        self.estimator.model.save(self.weightsDir+'lastEpoch.h5')

    def getFinalWeights(self):
        max_epoch = -1
        final_weights = 'lastEpoch.h5'
        for filename in os.listdir(self.weightsDir):
            if ".h5" and "model" in filename:
                this_epoch = filename.split(".")[1]
                if int(this_epoch) > max_epoch: 
                    final_weights = 'model.' + this_epoch + '.h5'
                    max_epoch = int(this_epoch)
        self.final_weights = final_weights
        self.max_epoch = max_epoch
        print("Final weights are from file {0} created after {1} epochs".format(self.final_weights, self.max_epoch))

    def trainNetwork(self):
        #get the inputs for the network and set the callbacks
        self.getInputs()
        self.setCallbacks()

        #load the data for the network and randomize it
        self.loadData()
        self.randomizeData()

        #create the model, train and save it
        self.createModel()
        self.fitModel()
        self.saveModel()
        self.getFinalWeights()

    def saveTrainInfo(self):
        plotMetrics.plotHistory(self.history, self.config['history_keys'], self.plotDir)
        self.config['input_dim'] = self.input_dim
        fout = open(self.filesDir + 'networkInfo.txt', 'w')
        fout.write('Datasets: ' + str(self.config['dataDir']) + 
                '\nFilters: ' + str(self.config['filters']) + 
                '\nBatch Size: ' + str(self.config['batch_size']) + 
                '\nBatch Norm: ' + str(self.config['batch_norm']) +  
                '\nInput Dim: ' + str(self.input_dim) + 
                '\nPatience Count: ' + str(self.config['patience_count']) + 
                '\nMetrics: ' + str(self.config['val_metrics']) + 
                '\nDeleted Elements: ' + str(self.config['delete_elements']) + 
                '\nSaved Tracks: ' + str(self.config['saveCategories']) + 
                '\nTrain Percentage: ' + str(self.config['trainPCT']) + 
                '\nVal Percentage: ' + str(self.config['valPCT']) + 
                '\nTotal Epochs: ' + str(self.max_epoch) + 
                '\nDropout: ' + str(self.config['dropout']) + 
                '\nMetrics: TP = %d, FP = %d, TN = %d, FN = %d' % (self.validator.classifications[0], self.validator.classifications[1], self.validator.classifications[2], self.validator.classifications[3]) + 
                '\nPrecision: ' + str(self.validator.classifications[4]) + 
                '\nRecall: ' + str(self.validator.classifications[5]))
        fout.close()

        utilities.saveConfig(self.config, self.filesDir+"config.json")

    def defineValidator(self):
        val_config = self.config
        val_config['weightsDir'] = self.weightsDir
        val_config['plotsName'] = 'validation_FakeProducer'
        val_args = utilities.validatorArgs(self.plotDir, '', self.weightsDir)

        self.validator = Validator(val_config, args=val_args)
        self.validator.loadModel(name=self.final_weights)

    def validateModel(self):
        self.defineValidator()
        self.validator.passData(self.testTracks, self.testEvents, self.testTruth)
        self.validator.makePredictions(inputs=self.inputs)
        self.validator.makePlots()

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('-o', '--outputDir', type=str, default='', help='Output directory to save output files and plots')
    parser.add_argument('-d', '--dataDir', type=str, default='', help='Input data directory')
    parser.add_argument('-c', '--config', type=str, default='', help='Config file of inputs for network (json)')
    parser.add_argument('-p', '--paramsFile', type=str, help='Parameters file') #TODO fix the help descriptions
    parser.add_argument('-i', '--index', type=int, help='Index for parameter')
    parser.add_argument('-g', '--grid', type=int, help='Number in grid search')
    parser.add_argument('-t', '--test', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    ################config parameters################
    dataDir = ["/store/user/mcarrigan/fakeTracks/converted_DYJets_aMCNLO_v9p3/", "/store/user/mcarrigan/fakeTracks/converted_NeutrinoGun_ext_v9p3/"]
    val_metrics = [keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
    batch_size = 64
    batch_norm = False
    epochs = 10
    filters=[12,8]
    input_dim = 178
    undersample = -1
    oversample = -1   
    dropout = 0.1
    delete_elements = ['passesSelection']
    saveCategories = [{'fake':True, 'real':True, 'pileup':False}, {'fake':True, 'real':False, 'pileup':False}]
    normalize_data = False
    patience_count = 20
    monitor = 'val_loss'
    trainPCT = 0.7
    valPCT = 0.5
    loadSplitDataset = False
    save_best_only = True
    mode = 'auto'
    threshold = 0.5
    history_keys = ['loss', 'auc_2', 'recall_2', 'precision_2']
    DEBUG = False

    #class_weights = False  
    #################################################

    config_dict = {}

    if args.config: config_dict = utilities.readConfig(configFile)
    else:
        config_dict = {'dataDir' : dataDir,
                    'val_metrics' : val_metrics,
                    'batch_size' : batch_size,
                    'batch_norm' : batch_norm,
                    'epochs' : epochs,
                    'filters' : filters,
                    'input_dim' : input_dim,
                    'undersample' : undersample,
                    'oversample' : oversample,
                    'dropout' : dropout,
                    'delete_elements' : delete_elements,
                    'saveCategories' : saveCategories,
                    'normalize_data' : normalize_data,
                    'patience_count' : patience_count,
                    'monitor' : monitor,
                    'trainPCT' : trainPCT,
                    'valPCT' : valPCT,
                    'loadSplitDataset' : loadSplitDataset,
                    'save_best_only' : save_best_only,
                    'mode' : mode,
                    'threshold' : threshold,
                    'history_keys' : history_keys,
                    'DEBUG' : DEBUG}
    
    myController = networkController(config_dict, args)
    myController.trainNetwork()
    myController.validateModel()
    myController.saveTrainInfo()

#####################################################
#####################################################


