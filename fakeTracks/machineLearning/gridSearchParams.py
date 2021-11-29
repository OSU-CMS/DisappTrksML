import os
import numpy as np
import utilities
import itertools as it
import sys


test_inputs = ['passesSelection', 'eventNumber', 'dEdxPixel', 'dEdxStrip', 'numMeasurementsPixel', 'numMeasurementsStrip', 'numSatMeasurementsPixel', 'numSatMeasurementsStrip', 'totalCharge', 'deltaRToClosestElectron', 'deltaRToClosestMuon', 'deltaRToClosestTauHad', 'normalizedChi2', 'layer1', 'charge1', 'subDet1', 'pixelHitSize1', 'pixelHitSizeX1', 'pixelHitSizeY1','stripSelection1', 'hitPosX1', 'hitPosY1', 'layer2', 'charge2', 'subDet2', 'pixelHitSize2', 'pixelHitSizeX2', 'pixelHitSizeY2', 'stripSelection2', 'hitPosX2', 'hitPosY2', 'layer3', 'charge3', 'subDet3', 'pixelHitSize3', 'pixelHitSizeX3', 'pixelHitSizeY3', 'stripSelection3', 'hitPosX3', 'hitPosY3', 'layer4', 'charge4', 'subDet4', 'pixelHitSize4', 'pixelHitSizeX4', 'pixelHitSizeY4', 'stripSelection4', 'hitPosX4', 'hitPosY4', 'layer5', 'charge5', 'subDet5', 'pixelHitSize5', 'pixelHitSizeX5', 'pixelHitSizeY5', 'stripSelection5', 'hitPosX5', 'hitPosY5', 'layer6', 'charge6', 'subDet6', 'pixelHitSize6', 'pixelHitSizeX6', 'pixelHitSizeY6', 'stripSelection6', 'hitPosX6', 'hitPosY6', 'layer7', 'charge7', 'subDet7', 'pixelHitSize7', 'pixelHitSizeX7', 'pixelHitSizeY7', 'stripSelection7', 'hitPosX7', 'hitPosY7', 'layer8', 'charge8', 'subDet8', 'pixelHitSize8', 'pixelHitSizeX8', 'pixelHitSizeY8', 'stripSelection8', 'hitPosX8', 'hitPosY8', 'layer9', 'charge9', 'subDet9', 'pixelHitSize9', 'pixelHitSizeX9', 'pixelHitSizeY9', 'stripSelection9', 'hitPosX9', 'hitPosY9', 'layer10', 'charge10', 'subDet10', 'pixelHitSize10', 'pixelHitSizeX10', 'pixelHitSizeY10', 'stripSelection10', 'hitPosX10', 'hitPosY10', 'layer11', 'charge11', 'subDet11', 'pixelHitSize11', 'pixelHitSizeX11', 'pixelHitSizeY11', 'stripSelection11', 'hitPosX11', 'hitPosY11', 'layer12', 'charge12', 'subDet12', 'pixelHitSize12', 'pixelHitSizeX12','pixelHitSizeY12', 'stripSelection12', 'hitPosX12', 'hitPosY12', 'layer13', 'charge13', 'subDet13', 'pixelHitSize13', 'pixelHitSizeX13', 'pixelHitSizeY13', 'stripSelection13', 'hitPosX13', 'hitPosY13', 'layer14', 'charge14', 'subDet14', 'pixelHitSize14', 'pixelHitSizeX14', 'pixelHitSizeY14', 'stripSelection14', 'hitPosX14', 'hitPosY14', 'layer15', 'charge15', 'subDet15', 'pixelHitSize15', 'pixelHitSizeX15', 'pixelHitSizeY15', 'stripSelection15', 'hitPosX15', 'hitPosY15', 'layer16', 'charge16', 'subDet16', 'pixelHitSize16', 'pixelHitSizeX16', 'pixelHitSizeY16', 'stripSelection16', 'hitPosX16', 'hitPosY16', 'sumEnergy', 'diffEnergy', 'dz1', 'd01', 'dz2', 'd02', 'dz3', 'd03']

class Parameters():

    def __init__(self, filters, batch_norm, undersampling, epochs, dataDir, input_dim, delete_elements, saveCategories, trainPCT, valPCT, loadSplitDataset, dropout):
        self.filters = filters
        self.batch_norm = batch_norm
        self.undersampling = undersampling
        self.epochs = epochs
        self.dataDir = dataDir
        self.input_dim = input_dim
        self.delete_elements = delete_elements
        self.saveCategories = saveCategories
        self.trainPCT = trainPCT
        self.valPCT = valPCT
        self.loadSplitDataset = loadSplitDataset
        self.dropout = dropout
        self.params = []

    def makeParamList(self, searchParam = 'default'):
        
        paramsList = []

        #if searchParam == 'default':
        self.params = [self.filters, 
                           self.batch_norm, 
                           self.undersampling, 
                           self.epochs, 
                           self.dataDir, 
                           self.input_dim, 
                           self.delete_elements, 
                           self.saveCategories, 
                           self.trainPCT, 
                           self.valPCT, 
                           self.loadSplitDataset, 
                           self.dropout] 

        if searchParam == 'default':
            paramsList = self.params

        elif searchParam == 'delete_elements':
            for x in self.params[6]:
                this_params = np.copy(self.params)
                this_params[6] = x
                print(this_params)
                paramsList.append(this_params)

        elif searchParam == 'filters':
            for x in self.params[0]:
                this_params = np.copy(self.params)
                this_params[0] = x
                print(this_params)
                paramsList.append(this_params)

        elif searchParam == 'dropout':
            print(self.__dict__[searchParam], type(self.__dict__[searchParam]))
            if(isinstance(self.__dict__[searchParam],list)==False):
                sys.exit("Need to provide list of dropout values to try")
                #print("")
            for x in self.__dict__[searchParam]:
                this_params = np.copy(self.params)
                this_params[11] = x
                print(this_params)
                paramsList.append(this_params)

        else:
            sys.exit("Search parameter is not regognized")

        self.params = paramsList


    def repeatSearch(self, repeat):
        #repeats list of parameters "repeat" number of times, used to average over many searches
        repeatedParams = []
        for param in self.params:
            for i in range(repeat):
                repeatedParams.append(param)

        self.params = repeatedParams

    def removeInputs(self):
        #function will remove each input from test_inputs and add all other inputs to delete_elements (repeated for all inputs in test_inputs)
        #used to test which inputs from test_inputs are useful
        delete_elements = []

        for ipar, par in enumerate(test_inputs):
            elements = np.copy(test_inputs)
            elements = elements[elements != par]
            delete_elements.append(elements)
            print(delete_elements[-1])

        self.delete_elements = delete_elements

    def filterSearchParams(self):

        if len(self.filters) > 2:
            print("Need to change length of prodcut to match filters")

        filterList = list(it.product(self.filters[0], self.filters[1]))

        self.filters = filterList

    def returnParams(self):
        params = np.array(self.params, dtype='object')
        return params


if __name__ == '__main__':

    repeat = 10

    #filters used for grid search
    #filters = [[32, 24, 16, 12], [24, 16, 12, 6]]
    filters = [32, 12]
    batch_norm = True
    undersampling = -1
    epochs = 100
    dataDir = ['/store/user/mcarrigan/fakeTracks/fakeNNInputDataset_DYJets.npz', '/store/user/mcarrigan/fakeTracks/fakeNNInputDataset_NG.npz']
    #dataDir = ["/store/user/mcarrigan/fakeTracks/converted_DYJets_aMCNLO_v9p2/", "/store/user/mcarrigan/fakeTracks/converted_NeutrinoGun_ext_v9p2/"]
    input_dim = 177
    #delete_elements = inputSearchParams()
    delete_elements = ['eventNumber', 'layer1', 'subDet1', 'stripSelection1', 'hitPosX1', 'hitPosY1','layer2', 'subDet2', 'stripSelection2', 'hitPosX2', 'hitPosY2',
                       'layer3', 'subDet3', 'stripSelection3', 'hitPosX3', 'hitPosY3', 'layer4', 'subDet4', 'stripSelection4', 'hitPosX4', 'hitPosY4',
                       'layer5', 'subDet5', 'stripSelection5', 'hitPosX5', 'hitPosY5', 'layer6', 'subDet6', 'stripSelection6', 'hitPosX6', 'hitPosY6',
                       'layer7', 'subDet7', 'stripSelection7', 'hitPosX7', 'hitPosY7', 'layer8', 'subDet8', 'stripSelection8', 'hitPosX8', 'hitPosY8',
                       'layer9', 'subDet9', 'stripSelection9', 'hitPosX9', 'hitPosY9', 'layer10', 'subDet10', 'stripSelection10', 'hitPosX10', 'hitPosY10',
                       'layer11', 'subDet11', 'stripSelection11', 'hitPosX11', 'hitPosY11', 'layer12', 'subDet12', 'stripSelection12', 'hitPosX12', 'hitPosY12',
                       'layer13', 'subDet13', 'stripSelection13', 'hitPosX13', 'hitPosY13', 'layer14', 'subDet14', 'stripSelection14', 'hitPosX14', 'hitPosY14',
                       'layer15', 'subDet15', 'stripSelection15', 'hitPosX15', 'hitPosY15', 'layer16', 'subDet16', 'stripSelection16', 'hitPosX16', 'hitPosY16']
    
    saveCategories = [{'fake':True, 'real':True, 'pileup':False}, {'fake':True, 'real':False, 'pileup':False}]
    trainPCT = 0.7
    valPCT = 0.5
    loadSplitDataset = True
    dropout = [0, 0.05, 0.1, 0.15, 0.2]
    #dropout = 0.2

    myparams = Parameters(filters, batch_norm, undersampling, epochs, dataDir, input_dim, delete_elements, saveCategories, trainPCT, valPCT, loadSplitDataset, dropout)

    #params = [filters, batch_norm, undersampling, epochs, dataDir, input_dim, delete_elements, saveCategories, trainPCT, valPCT, loadSplitDataset, dropout]
    #params = makeParams('filters', params)
    #params = np.array(params, dtype='object')
    #params = repeatSearch(repeat, params)


    #myparams.repeatSearch(10)
    #myparams.removeInputs()
    #myparams.filterSearchParams()

    myparams.makeParamList('dropout')
    myparams.repeatSearch(repeat)
    params = myparams.returnParams()

    print(params)
    np.save('params.npy', params)



    jobInfo = [repeat]
    np.save('jobInfo.npy', jobInfo)

