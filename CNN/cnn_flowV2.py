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


def build_model(input_shape = (40,40,3), layers=1,filters=64,opt='adadelta',kernels=(1,1),output_bias=0):
    
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
              metrics=['accuracy'])
    #print(model.summary())

    return model

  
# generate batches of images from files
class generator(keras.utils.Sequence):
  
  def __init__(self, filesE, filesBkg, batch_size, dataDir, e_fraction=-1):
    self.filesE = filesE
    self.filesBkg = filesBkg
    self.batch_size = batch_size
    self.dataDir = dataDir
    self.e_fraction = e_fraction
    
  def __len__(self) :
    return len(self.filesE)

  def __getitem__(self, idx) :

    filenamesE = self.filesE[idx]
    filenamesBkg = self.filesBkg[idx]

    # import images
    e_images = np.array([])
    for i,file in enumerate(filenamesE): 
      temp = np.load(self.dataDir+'e_'+tag1+str(file)+'.npz')['images']
      if(not(temp.shape[0] > 0)): print("FUCK",file)
      if(i==0): e_images = temp
      else: e_images = np.concatenate((e_images,temp))
    
    bkg_images = np.array([])
    for i,file in enumerate(filenamesBkg): 
      temp = np.load(self.dataDir+'bkg_'+tag2+str(file)+'.npz')['images']
      if(i==0): bkg_images = temp
      else: bkg_images = np.concatenate((bkg_images,temp))

    numE = e_images.shape[0]
    numBkg = bkg_images.shape[0]

    e_images = e_images[:,1:]
    added_e = []
    while((numE+len(added_e))*1.0 / batch_size < self.e_fraction):
      added_e.append(e_images[random.randint(0,numE-1),:])
    
    # number of added electrons
    numAdded = len(added_e)
    if(numAdded > 0): added_e = np.reshape(added_e,[numAdded, 6400])

    # fill the rest of the batch with random bkg images
    batch_indices = list(range(0,bkg_images.shape[0]))
    random.shuffle(batch_indices)
    bkg_images = bkg_images[batch_indices[(numE+numAdded):batch_size],1:]
    
    # join and reshape the images
    if(numAdded > 0): batch_x = np.concatenate((e_images,added_e,bkg_images))
    else: batch_x = np.concatenate((e_images,bkg_images))
    
    # debugging
    assert batch_x.shape[0] == batch_size, "batch_x doesn't match batch size"

    batch_x = np.reshape(batch_x,(batch_size,40,40,4))
    batch_x = batch_x[:,:,:,[0,2,3]]

    # labels
    batch_y = np.concatenate((np.ones(numE),np.ones(numAdded),np.zeros(numBkg)))
    batch_y = keras.utils.to_categorical(batch_y, 2)

    # shuffle the images and labels
    indices = list(range(len(batch_x)))
    random.shuffle(indices)
    batch_x = batch_x[indices]
    batch_y = batch_y[indices]

    return np.array(batch_x), np.array(batch_y)


if __name__ == "__main__":

  # limit CPU usage
  config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads = 2,   
                          intra_op_parallelism_threads = 2)
  tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

  dataDir = "/data/disappearingTracks/electron_selection_DYJetsToll_M50/"
  tag1 = '0p25_tanh_'
  tag2 = '0p25_tahn_'
  workDir = '/home/llavezzo/'
  plotDir = workDir + 'plots/cnn/'
  weightsDir = workDir + 'weights/cnn/'

  ################config parameters################
  pos,neg = 431, 91805              # FIXME: fix class weights, output bias
  batch_size = 256
  epochs = 1
  patience_count = 10
  maxFiles = 500
  val_size = 0.2
  img_rows, img_cols = 40, 40
  channels = 3
  input_shape = (img_rows,img_cols,channels)
  class_weights = False             # FIXME: not implemented yet
  e_fraction = 0.1
  #################################################

  if(not os.path.isdir(plotDir)): os.system('mkdir '+str(plotDir))
  if(not os.path.isdir(weightsDir)): os.system('mkdir '+str(weightsDir))

  """
  infos:

  0: ID
  1: matched track gen truth flavor (1: electrons, 2: muons, 0: everything else)
  2: nPV
  3: deltaRToClosestElectron
  4: deltaRToClosestMuon
  5: deltaRToClosestTauHaud

  """

  # import count dicts
  with open(dataDir+'eCounts.json') as json_file:
    eCounts = json.load(json_file)
  with open(dataDir+'bkgCounts.json') as json_file:
    bkgCounts = json.load(json_file)


  nTotE, nTotBkg = 10000,100000

  nBatches = (nTotE + nTotBkg)*1.0/batch_size
  nBatchE = int(nTotE/nBatches)
  nBatchBkg = int(nTotBkg/nBatches)

  filesE, filesBkg = [], []

  nSavedE = 0
  thisBatchE = 0
  temp = []
  for thisFile, thisFileE in eCounts.items():
    if(thisFileE < 1): continue
    if(nSavedE > nTotE): break

    temp.append(thisFile)
    thisBatchE += thisFileE
    
    if(thisBatchE >= nBatchE or thisFile == list(eCounts.keys())[-1]):
      nSavedE += thisBatchE
      filesE.append(temp)
      temp = []
      thisBatchE = 0
    

  nSavedBkg = 0
  thisBatchBkg = 0
  temp = []
  for thisFile, thisFileBkg in bkgCounts.items():
    if(thisFileBkg < 1): continue
    if(nSavedBkg > nTotBkg): break

    temp.append(thisFile)
    thisBatchBkg += thisFileBkg

    if(thisBatchBkg >= nBatchBkg or thisFile == list(bkgCounts.keys())[-1]):
      nSavedBkg += nBatchBkg
      filesBkg.append(temp)
      temp = []
      thisBatchBkg = 0

  print("Requested:")
  print("\t",nTotE,"electron events and",nTotBkg,"background events")
  print("Using:")
  print("\t",nSavedE,"electron events and",nSavedBkg,"background events")
  print("From total available events:")
  print("\t",sum(list(eCounts.values())),"electron events and",sum(list(bkgCounts.values())),"background events")

  if(nTotE > nSavedE): sys.exit("ERROR: Requested more electron events than are available")
  if(nTotBkg > nSavedBkg): sys.exit("ERROR: Requested more electron events than available")

  while(len(filesE) > len(filesBkg)):
    filesE = filesE[:-1]
  while(len(filesE) < len(filesBkg)):
    filesBkg = filesBkg[:-1]

  # split files into train and validation sets
  random.shuffle(filesE)
  random.shuffle(filesBkg)
  trainFilesE = filesE[:int((1-val_size)*len(filesE))]
  valFilesE = filesE[int((1-val_size)*len(filesE)):]
  trainFilesBkg = filesBkg[:int((1-val_size)*len(filesBkg))]
  valFilesBkg = filesBkg[int((1-val_size)*len(filesBkg)):]

  print("Training on",len(trainFilesE),"batches of files and validating on",len(valFilesE))

  # initialize generators
  train_generator = generator(trainFilesE, trainFilesBkg, batch_size, dataDir, e_fraction)
  val_generator = generator(valFilesE, valFilesBkg, batch_size, dataDir, e_fraction)

  # initialize output bias
  output_bias = np.log(pos/neg)       #FIXME: doesn't work with oversampling

  model = build_model(input_shape = input_shape, 
                      layers = 5, filters = 64, opt='adam',
                      output_bias=0)

  weightsFile = 'first_model'

  weight_for_0 = (1/neg)*(neg+pos)/2.0  #FIXME: doesn't work oversampling
  weight_for_1 = (1/pos)*(neg+pos)/2.0
  class_weights = {0: weight_for_0, 1: weight_for_1}
    
  callbacks = [
    keras.callbacks.EarlyStopping(patience=patience_count),
    keras.callbacks.ModelCheckpoint(filepath=weightsDir+weightsFile+'.h5',
                                    save_weights_only=True,
                                    monitor='val_loss',
                                    mode='auto',
                                    save_best_only=True),
  ]

  history = model.fit(train_generator, 
                      epochs = epochs,
                      verbose= 1,
                      validation_data=val_generator,
                      callbacks=callbacks)
                      #class_weight=class_weights)

  model.load_weights(weightsDir+weightsFile+'.h5')

  # predict each of the validation files individually
  predictions, true = [],[]
  for file in list(set(valFiles)):
    print(file)
    f1 = "0p25_tanh_"+str(file)+".npz"
    f2 = "0p25_tahn_"+str(file)+".npz"
    temp1 = np.load(dataDir+"e_"+f1)
    temp2 = np.load(dataDir+"bkg_"+f2) 
    images_e = temp1['images'][:,1:]
    images_bkg = temp2['images'][:,1:]
    images = np.concatenate((images_e,images_bkg))
    images = np.reshape(images, [len(images),40,40,4])
    x_test = images[:,:,:,[0,2,3]]
    y_test = np.concatenate((np.ones(len(images_e)),np.zeros(len(images_bkg))))
    y_test = keras.utils.to_categorical(y_test, num_classes=2)
    predictions.append(model.predict(x_test))
    true.append(y_test)

  
  utils.plot_history(history, plotDir,['loss','acc'])

  print()
  print("Calculating and plotting confusion matrix")
  cm = utils.calc_cm(y_test,predictions)
  utils.plot_confusion_matrix(cm,['bkg','e'],plotDir + 'cm.png')
  print()

  print("Plotting certainty")
  utils.plot_certainty(y_test,predictions,plotDir+'certainty.png')
  print()

  precision, recall = utils.calc_binary_metrics(cm)
  print("Precision = TP/(TP+FP) = fraction of predicted true actually true ",round(precision,3))
  print("Recall = TP/(TP+FN) = fraction of true class predicted to be true ",round(recall,3))
  auc = roc_auc_score(y_test,predictions)
  print("AUC Score:",round(auc,5))
  print()

  fileOut = open(plotDir+"metrics.txt","w")
  fileOut.write("Precision = TP/(TP+FP) = fraction of predicted true actually true "+str(round(precision,3))+"\n")
  fileOut.write("Recall = TP/(TP+FN) = fraction of true class predicted to be true "+str(round(recall,3))+"\n")
  fileOut.write("AUC Score:"+str(round(auc,5)))
  fileOut.close()