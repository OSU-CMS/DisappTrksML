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
    for i,file in enumerate(filenamesE): 
      temp = np.load(self.dataDir+'e_'+tag1+str(file)+'.npz')['images']
      if(i==0): e_images = temp
      else: e_images = np.concatenate((e_images,temp))
    
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

    # fill the rest of the batch with bkg images
    # random.shuffle(bkg_images)          #FIXME: could shuffle the background to have a different sample every epoch
    bkg_images = bkg_images[:(batch_size-(numE+numAdded)),1:]
    
    # join and reshape the images
    if(numAdded > 0): batch_x = np.concatenate((e_images,added_e,bkg_images))
    else: batch_x = np.concatenate((e_images,bkg_images))

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
  weightsFile = 'first_model'


  ################config parameters################
  """
  nTotE, nTotBkg:
    how many electrons, background events to use
    from maximum 15463 electron events and 3256305 background events
  e_fraction:
    use to oversample electron events,
    fraction of electron events per batch
  """
  nTotE, nTotBkg = 15000,200000
  batch_size = 256
  epochs = 10
  patience_count = 10
  val_size = 0.2
  img_rows, img_cols = 40, 40
  channels = 3
  input_shape = (img_rows,img_cols,channels)
  class_weights = True             
  e_fraction = 0.2              
  #################################################


  if(not os.path.isdir(plotDir)): os.system('mkdir '+str(plotDir))
  if(not os.path.isdir(weightsDir)): os.system('mkdir '+str(weightsDir))

  # import count dicts
  with open(dataDir+'eCounts.json') as json_file:
    eCounts = json.load(json_file)
  with open(dataDir+'bkgCounts.json') as json_file:
    bkgCounts = json.load(json_file)

  # can't request more events than we have
  if(nTotE > sum(list(eCounts.values()))): sys.exit("ERROR: Requested more electron events than are available")
  if(nTotBkg > sum(list(bkgCounts.values()))): sys.exit("ERROR: Requested more electron events than available")

  # batches per epoch, number of electrons, bkg events per batch
  nBatches = np.ceil((nTotE + nTotBkg)*1.0/batch_size)
  nElectronsPerBatch = int(nTotE/nBatches)
  nBkgPerPatch = int(nTotBkg/nBatches)

  # create batches of files to reach in each batch
  # the desired number of events of each class
  filesE, filesBkg = [], []
  
  nSavedE = 0
  thisBatchE = 0
  temp = []
  keys = list(eCounts.keys())
  while nSavedE < nTotE:
    
    random.shuffle(keys)
    for thisFile in keys:
      thisFileE = eCounts[thisFile]
      if(thisFileE < 1): continue
      if(nSavedE > nTotE): break

      temp.append(thisFile)
      thisBatchE += thisFileE
      
      if(thisBatchE >= nElectronsPerBatch or thisFile == list(eCounts.keys())[-1]):
        nSavedE += nElectronsPerBatch
        filesE.append(temp)
        temp = []
        thisBatchE = 0


  nSavedBkg = 0
  thisBatchBkg = 0
  temp = []
  keys = list(bkgCounts.keys())
  random.shuffle(keys)
  for thisFile in keys:
    thisFileBkg = bkgCounts[thisFile]
    if(thisFileBkg < 1): continue
    if(nSavedBkg > nTotBkg): break

    temp.append(thisFile)
    thisBatchBkg += thisFileBkg

    if(thisBatchBkg >= nBkgPerPatch or thisFile == list(bkgCounts.keys())[-1]):
      nSavedBkg += nBkgPerPatch
      filesBkg.append(temp)
      temp = []
      thisBatchBkg = 0

  print("Requested:")
  print("\t",nTotE,"electron events and",nTotBkg,"background events")
  print("Using:")
  print("\t",nSavedE,"electron events and",nSavedBkg,"background events")
  print("From total available events:")
  print("\t",sum(list(eCounts.values())),"electron events and",sum(list(bkgCounts.values())),"background events")

  
  # make sure they are same length
  while(len(filesE) > len(filesBkg)):
    filesE = filesE[:-1]
  while(len(filesE) < len(filesBkg)):
    filesBkg = filesBkg[:-1]

  # split batches of files into train and validation sets
  random.shuffle(filesE)
  random.shuffle(filesBkg)
  trainFilesE = filesE[:int((1-val_size)*len(filesE))]
  valFilesE = filesE[int((1-val_size)*len(filesE)):]
  trainFilesBkg = filesBkg[:int((1-val_size)*len(filesBkg))]
  valFilesBkg = filesBkg[int((1-val_size)*len(filesBkg)):]

  print("Training on:")
  print("\t",len(trainFilesE),"batches of files (approx.",nElectronsPerBatch*len(trainFilesE),"electron and",nBkgPerPatch*len(trainFilesE), "background events)")
  print("Validating on:")
  print("\t",len(valFilesE),"batches of files (approx.",nElectronsPerBatch*len(valFilesE),"electron and",nBkgPerPatch*len(valFilesE)," background events)")

  # initialize generators
  train_generator = generator(trainFilesE, trainFilesBkg, batch_size, dataDir, e_fraction)
  val_generator = generator(valFilesE, valFilesBkg, batch_size, dataDir, e_fraction)

  # initialize output bias
  if(e_fraction == -1): output_bias = np.log(nTotE/nTotBkg)
  else: output_bias = np.log(1.0*e_fraction/(1-e_fraction))

  model = build_model(input_shape = input_shape, 
                      layers = 5, filters = 64, opt='adam',
                      output_bias=output_bias)

  # class weights
  if(e_fraction == -1):
    weight_for_0 = (1/nTotBkg)*(nTotBkg+nTotE)/2.0
    weight_for_1 = (1/nTotE)*(nTotBkg+nTotE)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
  else:
    weight_for_0 = 1/(2*(1-e_fraction))  #FIXME: check if this works
    weight_for_1 = 1/(2.0*e_fraction)
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
                        verbose= 2,
                        validation_data=val_generator,
                        callbacks=callbacks,
                        class_weight=class_weight)
  else:
    history = model.fit(train_generator, 
                        epochs = epochs,
                        verbose= 2,
                        validation_data=val_generator,
                        callbacks=callbacks)
                        
  model.load_weights(weightsDir+weightsFile+'.h5')

  # predict each of the validation files individually
  valFilesE = [file for batch in valFilesE for file in batch]
  valFilesBkg = [file for batch in valFilesBkg for file in batch]
  for i,file in enumerate(valFilesE):
    fname = "e_"+tag1+file+".npz"
    temp = np.load(dataDir+fname) 
    images = temp['images'][:,1:]
    images = np.reshape(images, [len(images),40,40,4])
    x_test = images[:,:,:,[0,2,3]]
    if(i==0): predictionsE = model.predict(x_test)
    else: predictionsE = np.concatenate([predictionsE, model.predict(x_test)])
  for i,file in enumerate(valFilesBkg):
    fname = "bkg_"+tag2+file+".npz"
    temp = np.load(dataDir+fname) 
    images = temp['images'][:batch_size,1:]
    images = np.reshape(images, [len(images),40,40,4])
    x_test = images[:,:,:,[0,2,3]]
    if(i == 0): predictionsB = model.predict(x_test)
    else: predictionsB = np.concatenate([predictionsB, model.predict(x_test)])
  
  predictions = np.concatenate((predictionsE, predictionsB))
  true = np.concatenate((np.ones(len(predictionsE)), np.zeros(len(predictionsB))))
  y_test = keras.utils.to_categorical(true, num_classes=2)


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