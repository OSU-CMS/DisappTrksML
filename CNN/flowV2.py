import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import utils
import json
import random
import sys

def build_model(input_shape = (40,40,3), layers=1,filters=64,opt='adadelta',kernels=(1,1),output_bias=0,metrics=['accuracy']):
    
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    for _ in range(layers-1):
        model.add(keras.layers.Conv2D(filters, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)))
        if(_%2 == 0):
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(2, activation='softmax',bias_initializer=keras.initializers.Constant(output_bias)))
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=metrics)
    #print(model.summary())
    
    return model

  
# generate batches of images from files
class generator(keras.utils.Sequence):
  
  def __init__(self, list_IDs, dataDir, batch_size=32, dim=6400, n_channels=4, n_classes=2, shuffle = True):
    self.list_IDs = list_IDs
    self.batch_size = batch_size
    self.dataDir = dataDir
    self.dim = dim
    self.n_channels = n_channels
    self.shuffle = shuffle
    self.n_classes = n_classes
    self.on_epoch_end()
    
  def on_epoch_end(self):
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
        np.random.shuffle(self.indexes)

  def __len__(self) :
    return int(np.floor(len(self.list_IDs)*1.0 / self.batch_size))

  def __data_generation(self, list_IDs_temp):

    # Initialization
    X = np.empty((self.batch_size, self.dim))
    y = np.empty((self.batch_size), dtype=int)

    # Generate data
    for i, ID in enumerate(list_IDs_temp):

        if(ID[0]=="0"): fname = 'bkg_'+tag+str(int(ID[1:5]))+'.npy'
        else: fname = 'e_'+tag+str(int(ID[1:5]))+'.npy'

        # Store sample
        X[i,] = np.load(dataDir+fname, mmap_mode='r')[int(ID[5:]),1:]

        # Store class
        y[i] = int(ID[0])
      
    X = np.reshape(X, (self.batch_size, 40,40,4))
    X = X[:,:,:,[0,2,3]]

    return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

  def __getitem__(self, index):

    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    # Find list of IDs
    list_IDs_temp = [self.list_IDs[k] for k in indexes]

    # Generate data
    X, y = self.__data_generation(list_IDs_temp)

    return X, y    


if __name__ == "__main__":

  # limit CPU usage
  config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads = 2,   
                          intra_op_parallelism_threads = 2)
  tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

  dataDir = "/data/disappearingTracks/electron_selectionV2/"
  tag = '0p25_tanh_'
  workDir = '/home/llavezzo/'
  plotDir = workDir + 'plots/cnn/'
  weightsDir = workDir + 'weights/cnn/'
  weightsFile = 'cnn_first_model'

  ################config parameters################
  nTrainE = 5000
  nValE = 1000
  undersample_bkg = 0.9          
  oversample_e = -1

  batch_size = 256
  epochs = 20
  patience_count = 10
  class_weights = True
  metrics = ['accuracy', 'Precision', 'Recall']

  img_rows, img_cols = 40, 40
  channels = 3
  input_shape = (img_rows,img_cols,channels)
  #################################################


  if(not os.path.isdir(plotDir)): os.system('mkdir '+str(plotDir))
  if(not os.path.isdir(weightsDir)): os.system('mkdir '+str(weightsDir))

  # load list of images
  imageList = np.load("imageList.npy")
  random.shuffle(imageList)

  # import count dicts
  with open(dataDir+'eCounts.json') as json_file:
    eCounts = json.load(json_file)
  with open(dataDir+'bkgCounts.json') as json_file:
    bkgCounts = json.load(json_file)

  # count how many events are in the files for each class
  availableE = sum(list(eCounts.values()))
  availableBkg = sum(list(bkgCounts.values()))

  # fractions for each class for the total dataset
  fE = availableE*1.0/(availableE + availableBkg)
  fBkg = availableBkg*1.0/(availableE + availableBkg)

  # calculate how much backgorund is desired to mantain
  # correct proportions
  nTrainBkg = int(np.ceil(nTrainE*fBkg/fE))
  if(undersample_bkg != -1): nTrainBkgUndersampled = int(np.ceil(nTrainE*undersample_bkg/(1-undersample_bkg)))
  nValBkg = int(np.ceil(nValE*fBkg/fE))

  # how many electrons to oversample
  if(oversample_e != -1):
    if(undersample_bkg == -1):
      nTrainEOversampled = int(np.ceil(nTrainBkg*oversample_e/(1-oversample_e)))
    else:
      nTrainEOversampled = int(np.ceil(nTrainBkgUndersampled*oversample_e/(1-oversample_e)))
    nTrainEToOversample = nTrainEOversampled-nTrainE
  
  # can't request more events than we have
  nTotE = nTrainE+nValE
  nTotBkg = nTrainBkg+nValBkg
  if(nTotE > availableE): sys.exit("ERROR: Requested more electron events than are available")
  if(nTotBkg > availableBkg): sys.exit("ERROR: Requested more electron events than available")

  # split images IDs by class
  eList, bkgList = [], []
  for img in imageList:
    if img[0] == "1":
      eList.append(img)
    if img[0] == "0":
      bkgList.append(img)

  # train/validation split
  trainE = eList[:nTrainE]
  if(oversample_e != -1):
    trainEOversampled = random.sample(trainE, nTrainEToOversample)
    trainE = np.concatenate((trainE, trainEOversampled))
  valE = eList[nTrainE:(nTrainE+nValE)]
  if(undersample_bkg == -1):
    trainBkg = bkgList[:nTrainBkg]
    valBkg = bkgList[nTrainBkg:(nTrainBkg+nValBkg)]
  else:
    trainBkg = bkgList[:nTrainBkgUndersampled]
    valBkg = bkgList[nTrainBkgUndersampled:(nTrainBkgUndersampled+nValBkg)]
  trainIDs = np.concatenate((trainE, trainBkg))
  valIDs = np.concatenate((valE, valBkg))
  random.shuffle(trainIDs)
  random.shuffle(valIDs)

  print("\t\tElectrons\tBackground\te/(e+bkg)")
  print("Requested:\t"+str(nTotE)+"\t\t"+str(nTotBkg)+"\t\t"+str(round(nTotE*1.0/(nTotE+nTotBkg),5)))
  if(undersample_bkg == -1):
    print("Training on:\t"+str(nTrainE)+"\t\t"+str(nTrainBkg)+"\t\t"+str(round(nTrainE*1.0/(nTrainE+nTrainBkg),5)))
  else:
    print("Training on:")
    print("Undersampling:\t"+str(nTrainE)+"\t\t"+str(nTrainBkgUndersampled)+"\t\t"+str(round(nTrainE*1.0/(nTrainE+nTrainBkgUndersampled),5)))
  if(oversample_e != -1 and undersample_bkg == -1):
    print("Oversampling:\t"+str(nTrainEOversampled)+"\t\t"+str(nTrainBkg)+"\t\t"+str(round(nTrainEOversampled*1.0/(nTrainEOversampled+nTrainBkg),5)))
  if(oversample_e != -1 and undersample_bkg != -1):
    print("Oversampling:\t"+str(nTrainEOversampled)+"\t\t"+str(nTrainBkgUndersampled)+"\t\t"+str(round(nTrainEOversampled*1.0/(nTrainEOversampled+nTrainBkgUndersampled),5)))
  print("Validating on:\t"+str(nValE)+"\t\t"+str(nValBkg)+"\t\t"+str(round(nValE*1.0/(nValE+nValBkg),5)))
  print("Dataset:\t"+str(availableE)+"\t\t"+str(availableBkg)+"\t\t"+str(round(fE,5)))
 
  # initialize generators
  # if oversampling in training data, set appropriate e_fraction in each batch
  train_generator = generator(trainIDs, dataDir, batch_size)
  val_generator = generator(valIDs, dataDir, batch_size)

  # initialize output bias
  if(undersample_bkg == -1 and oversample_e == -1): 
    output_bias = np.log(nTotE/nTotBkg)
  elif(undersample_bkg != -1 and oversample_e == -1): 
    output_bias = np.log(1.0*1/(undersample_bkg/(1-undersample_bkg)))
  else: 
    output_bias = np.log(1.0*oversample_e/(1-oversample_e))

  model = build_model(input_shape = input_shape, 
                      layers = 5, filters = 64, opt='adam',
                      output_bias=output_bias,
                      metrics=metrics)

  # class weights
  if(undersample_bkg == -1 and oversample_e == -1): 
    weight_for_0 = (1/nTotBkg)*(nTotBkg+nTotE)/2.0
    weight_for_1 = (1/nTotE)*(nTotBkg+nTotE)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
  elif(undersample_bkg != -1 and oversample_e == -1): 
    weight_for_0 = 1/(2*undersample_bkg)
    weight_for_1 = 1/(2.0*(1-undersample_bkg))
    class_weight = {0: weight_for_0, 1: weight_for_1}
  elif(oversample_e != -1): 
    weight_for_0 = 1/(2*(1-oversample_e))
    weight_for_1 = 1/(2.0*oversample_e)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
  callbacks = [
    keras.callbacks.EarlyStopping(patience=patience_count),
    keras.callbacks.ModelCheckpoint(filepath=weightsDir+weightsFile+'.h5',
                                    save_weights_only=True,
                                    monitor='val_loss',
                                    mode='auto',
                                    save_best_only=True),
  ]

  if(class_weights):
    history = model.fit(train_generator, 
                        epochs = epochs,
                        verbose= 1,
                        validation_data=val_generator,
                        callbacks=callbacks,
                        class_weight=class_weight)
  else:
    history = model.fit(train_generator, 
                        epochs = epochs,
                        verbose= 1,
                        validation_data=val_generator,
                        callbacks=callbacks)
                        
  model.load_weights(weightsDir+weightsFile+'.h5')

  true = []
  predictions = []
  for ID in valIDs:

    if(ID[0]=="0"): fname = 'bkg_'+tag+str(int(ID[1:5]))+'.npy'
    else: fname = 'e_'+tag+str(int(ID[1:5]))+'.npy'
    temp = np.load(dataDir+fname, mmap_mode='r')[int(ID[5:]),1:]  

    true.append(int(ID[0]))
    predictions.append(model.predict(temp))
  
  true = keras.utils.to_categorical(true, num_classes=2)

  utils.plot_history(history, plotDir,['loss','accuracy'])

  print()
  print("Calculating and plotting confusion matrix")
  cm = utils.calc_cm(y_test,predictions)
  utils.plot_confusion_matrix(cm,['bkg','e'],plotDir + 'cm.png')
  print()

  print("Plotting certainty")
  utils.plot_certainty(y_test,predictions,plotDir+'certainty.png')
  print()

  precision, recall = utils.calc_binary_metrics(cm)
  print("Precision = TP/(TP+FP) = fraction of predicted true actually true ",round(precision,5))
  print("Recall = TP/(TP+FN) = fraction of true class predicted to be true ",round(recall,5))
  auc = roc_auc_score(y_test,predictions)
  print("AUC Score:",round(auc,5))
  print()

  fileOut = open(plotDir+"metrics.txt","w")
  fileOut.write("Precision = TP/(TP+FP) = fraction of predicted true actually true "+str(round(precision,5))+"\n")
  fileOut.write("Recall = TP/(TP+FN) = fraction of true class predicted to be true "+str(round(recall,5))+"\n")
  fileOut.write("AUC Score:"+str(round(auc,5)))
  fileOut.close()
  print("Wrote out metrics to metrics.txt")