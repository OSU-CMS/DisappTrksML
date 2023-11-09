#!/usr/bin/env python3
""" This module is used to control a general neural network """

from abc import ABC, abstractmethod

class NetworkController(ABC):
    """
    This abstract class requires that the following methods be defined in your network:
    buildModel() - responsible for creating layers, nodes, and activation functions in model

    Your model should also inherit this class

    """
   def __init__(self, model)
        self.config = config
        self.args = args
        self.model = model

        self.initialize()

    def initialize(self):
        if args.gpu:
           self.gpu_settings()
        else:
            self.cpu_settings()

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

        # suppress warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    @abstractmethod
    def buildModel(self):
        raise NotImplementedError("Subclasses must implement buildModel()")

    @abstractmethod
    def trainNetwork(self):
        raise NotImplementedError("Subclasses must implement trainNetwork()")



    def initialize(self):
        if args.gpu:
            self.gpu_settings()
        else:
            self.cpu_settings()

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
            print("testing output directory name", networkController.outputDir)
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

        # suppress warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    def gpu_settings(self):
        config=tf.compat.v1.ConfigProto(log_device_placement=True)
        sess = tf.compat.v1.Session(config=config)

        # suppress warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

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
