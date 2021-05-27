import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
  
  def __init__(self, filesE, filesBkg, batch_size, dataDir, nElectronsPerBatch):
    self.filesE = filesE
    self.filesBkg = filesBkg
    self.batch_size = batch_size
    self.dataDir = dataDir
    self.nElectronsPerBatch = nElectronsPerBatch
    
  def __len__(self) :
    return len(self.filesE)

  def __getitem__(self, idx) :

    filenamesE = self.filesE[idx]
    filenamesBkg = self.filesBkg[idx]

    # import images
    for i,file in enumerate(filenamesE): 
      temp = np.load(self.dataDir+'e_'+tag+str(file)+'.npy')
      if(i==0): e_images = temp
      else: e_images = np.concatenate((e_images,temp))
    for i,file in enumerate(filenamesBkg): 
      temp = np.load(self.dataDir+'bkg_'+tag+str(file)+'.npy')
      if(i==0): bkg_images = temp
      else: bkg_images = np.concatenate((bkg_images,temp))

    numE = e_images.shape[0]
    numBkg = bkg_images.shape[0]

    # shuffle and select appropriate amount of electrons, bkg
    indices = list(range(e_images.shape[0]))
    random.shuffle(indices)
    e_images = e_images[indices[:self.nElectronsPerBatch],1:]

    indices = list(range(bkg_images.shape[0]))
    random.shuffle(indices)
    bkg_images = bkg_images[indices[:(batch_size-self.nElectronsPerBatch)],1:]

    # concatenate images and suffle them, create labels
    batch_x = np.concatenate((e_images,bkg_images))
    batch_y = np.concatenate((np.ones(self.nElectronsPerBatch),np.zeros(batch_size-self.nElectronsPerBatch)))
    
    indices = list(range(batch_x.shape[0]))
    random.shuffle(indices)

    batch_x = batch_x[indices[:batch_size],:]
    batch_x = np.reshape(batch_x,(batch_size,40,40,4))
    batch_x = batch_x[:,:,:,[0,2,3]]

    batch_y = batch_y[indices[:batch_size]]
    batch_y = keras.utils.to_categorical(batch_y, 2)

    return np.array(batch_x), np.array(batch_y)

# generate batches of images from files
class val_generator(keras.utils.Sequence):
  
  def __init__(self, filesE, filesBkg, dataDir, batch_size=-1):
    self.filesE = filesE
    self.filesBkg = filesBkg
    self.dataDir = dataDir
    self.batch_size = batch_size
    
  def __len__(self) :
    return len(self.filesE)

  def __getitem__(self, idx) :

    fileE = self.filesE[idx]
    fileBkg = self.filesBkg[idx]

    e_images = np.load(self.dataDir+'e_'+tag+str(fileE)+'.npy')
    bkg_images = np.load(self.dataDir+'bkg_'+tag+str(fileBkg)+'.npy')
    random.shuffle(e_images)
    random.shuffle(bkg_images)

    nLoadedE = e_images.shape[0]
    nLoadedBkg = bkg_images.shape[0]

    f = nLoadedE*1.0/(nLoadedE+nLoadedBkg)
    if(self.batch_size != -1):
      nE = int(np.ceil(f*self.batch_size))
      nBkg = int(self.batch_size - nE)
    else:
      nE = int(nLoadedE)
      nBkg = int(nLoadedBkg)

    # concatenate images and suffle them, create labels
    batch_x = np.concatenate((e_images[:nE],bkg_images[:nBkg]))
    batch_y = np.concatenate((np.ones(nE),np.zeros(nBkg)))
    
    indices = list(range(batch_x.shape[0]))
    random.shuffle(indices)

    batch_x = batch_x[indices,1:]
    batch_x = np.reshape(batch_x,(batch_x.shape[0],40,40,4))
    batch_x = batch_x[:,:,:,[0,2,3]]

    batch_y = batch_y[indices]
    batch_y = keras.utils.to_categorical(batch_y, 2)

    return np.array(batch_x), np.array(batch_y)


if __name__ == "__main__":

  # limit CPU usage
  # config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads = 2,   
  #                         intra_op_parallelism_threads = 2)
  # tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

  dataDir = "/data/disappearingTracks/electron_selectionV2/"
  tag = '0p25_tanh_'
  workDir = '/home/llavezzo/'
  plotDir = workDir + 'plots/cnn/'
  weightsDir = workDir + 'weights/cnn/'
  weightsFile = 'flow_model'

  ################config parameters################
  """
  nTotE:
    how many electron events to use
    from maximum 15463 electron events and 3256305 background events
    IMPORTANT: the validation set will use nTotE * val_size electrons
    but as many background events as needed to keep the ratio between the classes
    equal to the one in the real data
  oversample_e:
    use to oversample electron events, fraction of electron events per batch
    set to -1 if it's not needed
  undersample_bkg:
    what fraction of train events to be bkg,
    set to -1 if it's not needed
  """
  train = False
  validate = True
  trainFile = "trainBatches"
  valFile = "valBatches"

  nTotE = 10000
  val_size = 0.2
  undersample_bkg = -1          
  oversample_e = -1   

  batch_size = 256
  epochs = 20
  patience_count = 5
  class_weights = True  
  metrics = ['accuracy', 'Precision', 'Recall']

  img_rows, img_cols = 40, 40
  channels = 3
  input_shape = (img_rows,img_cols,channels)
  #################################################


  if(not os.path.isdir(plotDir)): os.system('mkdir '+str(plotDir))
  if(not os.path.isdir(weightsDir)): os.system('mkdir '+str(weightsDir))

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
  
  # calculate how many total background events for the requested electrons
  # to keep the same fraction of events, or under sample
  nTotBkg = int(nTotE*1.0*availableBkg/availableE)
  if(undersample_bkg!=-1): nTotBkg = int(nTotE*1.0*undersample_bkg/(1-undersample_bkg))

  # can't request more events than we have
  if(nTotE > availableE): sys.exit("ERROR: Requested more electron events than are available")
  if(nTotBkg > availableBkg): sys.exit("ERROR: Requested more electron events than available")

  # batches per epoch, number of electrons/bkg events per batch
  nBatches = int(np.ceil((nTotE + nTotBkg)*1.0/batch_size))
  nElectronsPerBatch = int(np.ceil(nTotE/nBatches))
  nBkgPerBatch = int(batch_size - nElectronsPerBatch)
  nBkgPerBatchData = int(nElectronsPerBatch * fBkg / fE)

  # create batches of files to reach in each batch
  # the desired number of events of each class
  filesE, filesBkg = [], []
  
  savedBatches = 0
  thisBatchE = 0
  temp = []
  keys = list(eCounts.keys())
  random.shuffle(keys)

  batchesE, indicesE = [],[]
  lastFile = 0
  lastEvent = 0
  last_iFile = 0
  for iBatch in range(nBatches):
    if(iBatch > 20): break
    print("iBatch",iBatch)
    batch = []
    indices = []
    thisBatchEvents = 0
    firstFile = lastFile
    firstEvent = lastEvent
    batch_completed = False
    for iFile, file in enumerate(keys):
      if(iFile < last_iFile): continue
      thisFileEvents = eCounts[file]
      for event in range(lastEvent,thisFileEvents):
        if(event < firstEvent): continue
        thisBatchEvents+=1
        if(thisBatchEvents==nElectronsPerBatch):
          last_iFile = iFile
          lastFile = file
          lastEvent = event
          batch.append(file)
          indices.append([firstEvent,lastEvent])
          batch_completed = True 
        if(event == thisFileEvents-1):
          firstEvent = 0
          lastEvent = event
          batch.append(file)
          indices.append([firstEvent,lastEvent]) 
        if(batch_completed): break
      if(batch_completed): break
    print(batch)
    print(indices) 
    batchesE.append(batch)
    indicesE.append(indices)

  tot = 0
  for batch, batch_indices in zip(batchesE,indicesE):
    thisBatch = 0
    for file, indices in zip(batch, batch_indices):
      tot+=indices[1]-indices[0]
      thisBatch+=indices[1]-indices[0]
    assert thisBatch == nElectronsPerBatch, thisBatch
  print(tot)

  events, files = [], []
  for file, events in eCounts.items():
    for evt in range(events):
      events.append(evt)
      files.append(file)
  for _ in nBatches:
    events[i,j]
    files[i,j]

    
  sys.exit(0)

  for thisFile in keys:
    thisFileE = eCounts[thisFile]
    if(thisFileE < 1): continue
    if(savedBatches > nBatches): break

    temp.append(thisFile)
    thisBatchE += thisFileE
    
    if(thisBatchE >= nElectronsPerBatch):
      filesE.append(temp)
      temp = []
      thisBatchE = 0
      savedBatches+=1

  random.shuffle(filesE)
  trainBatchesE = filesE[:int((1-val_size)*len(filesE))]
  valBatchesE = filesE[int((1-val_size)*len(filesE)):]

  trainFilesE = [file for batch in trainBatchesE for file in batch]
  valFilesE = [file for batch in valBatchesE for file in batch]
  nTrainE = int(sum(eCounts[file] for file in trainFilesE))
  nValE = int(sum(eCounts[file] for file in valFilesE))
  nSavedETrain = int(nElectronsPerBatch*len(trainBatchesE))
  nSavedEVal = int(nElectronsPerBatch*len(valBatchesE))
  
  keys = list(bkgCounts.keys())
  random.shuffle(keys)
  trainBatchesBkg, valBatchesBkg = [], []
  for i in range(len(trainBatchesE)):
    iFile = 0
    thisBatchBkg = 0
    batch = []
    while(thisBatchBkg < nBkgPerBatch):
      thisFile = keys[iFile]
      thisFileBkg = bkgCounts[thisFile]
      if(thisFileBkg < 1): continue

      batch.append(thisFile)
      thisBatchBkg += thisFileBkg

      iFile+=1 

    trainBatchesBkg.append(batch)
      
  for i in range(len(valBatchesE)):
    iFile = 0
    thisBatchBkg = 0
    batch = []
    while(thisBatchBkg < nBkgPerBatchData):
      thisFile = keys[iFile]
      thisFileBkg = bkgCounts[thisFile]
      if(thisFileBkg < 1): continue

      batch.append(thisFile)
      thisBatchBkg += thisFileBkg

      iFile+=1 
    
    valBatchesBkg.append(batch)
  
  trainFilesBkg = [file for batch in trainBatchesBkg for file in batch]
  valFilesBkg = [file for batch in valBatchesBkg for file in batch]
  nTrainBkg = int(sum(bkgCounts[file] for file in trainFilesBkg))
  nValBkg = int(sum(bkgCounts[file] for file in valFilesBkg))
  nSavedBkgTrain = int(nBkgPerBatch*len(trainBatchesBkg))
  nSavedBkgVal = int(nBkgPerBatchData*len(valBatchesBkg))

  batch_size_val = int(nBkgPerBatchData+nElectronsPerBatch)

  print("\t\tElectrons\tBackground\te/(e+bkg)")
  print("Requested:\t"+str(nTotE)+"\t\t"+str(nTotBkg)+"\t\t"+str(round(nTotE*1.0/(nTotE+nTotBkg),3)))
  print("Training on:\t"+str(nSavedETrain)+"\t\t"+str(nSavedBkgTrain)+"\t\t"+str(round(nSavedETrain*1.0/(nSavedETrain+nSavedBkgTrain),3)))
  print("Validating on:\t"+str(nSavedEVal)+"\t\t"+str(nSavedBkgVal)+"\t\t"+str(round(nSavedEVal*1.0/(nSavedEVal+nSavedBkgVal),3)))
  print("Dataset:\t"+str(availableE)+"\t\t"+str(availableBkg)+"\t\t"+str(round(fE,3)))

  # oversample the training electron files if oversample_e != -1
  nElectronsPerBatchOversampled = int(np.ceil(batch_size*oversample_e))
  ovsFiles = list([file for batch in trainBatchesE for file in batch])
  random.shuffle(ovsFiles)
  for i,batch in enumerate(trainBatchesE):
    nElectronsThisBatch = 0
    for file in batch: nElectronsThisBatch+=eCounts[file]
    while nElectronsThisBatch < nElectronsPerBatchOversampled:
      randFile = ovsFiles[random.randint(0,len(ovsFiles)-1)]
      trainBatchesE[i].append(randFile)
      nElectronsThisBatch += eCounts[randFile]
  if(oversample_e != -1):
    print("Oversampling:")
    print("\t Number of electrons per batch:",nElectronsPerBatchOversampled)
    print("\t",len(trainBatchesE),"batches of files (approx.",nElectronsPerBatchOversampled*len(trainBatchesE),"electron and",(batch_size-nElectronsPerBatchOversampled)*len(trainBatchesE), "background events)")

  # initialize generators
  # if oversampling in training data, set appropriate oversample_e in each batch
  if(oversample_e == -1):  train_generator = generator(trainBatchesE, trainBatchesBkg, batch_size, dataDir, nElectronsPerBatch)
  else: train_generator = generator(trainBatchesE, trainBatchesBkg, batch_size, dataDir, nElectronsPerBatchOversampled)
  val_generator = generator(valBatchesE, valBatchesBkg, batch_size_val, dataDir, nElectronsPerBatch)

  # initialize output bias
  if(oversample_e == -1): output_bias = np.log(nTotE/nTotBkg)
  else: output_bias = np.log(1.0*oversample_e/(1-oversample_e))

  model = build_model(input_shape = input_shape, 
                      layers = 5, filters = 64, opt='adam',
                      output_bias=output_bias,
                      metrics=metrics)

  # class weights
  if(oversample_e == -1):
    weight_for_0 = (1/nTotBkg)*(nTotBkg+nTotE)/2.0
    weight_for_1 = (1/nTotE)*(nTotBkg+nTotE)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
  else:
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

  if(train and class_weights):
    history = model.fit(train_generator, 
                        epochs = epochs,
                        verbose= 1,
                        validation_data=val_generator,
                        callbacks=callbacks,
                        class_weight=class_weight)
  elif(train):
    history = model.fit(train_generator, 
                        epochs = epochs,
                        verbose= 1,
                        validation_data=val_generator,
                        callbacks=callbacks)
                        
  model.load_weights(weightsDir+weightsFile+'.h5')

  if(train):
    model.save_weights(weightsDir+weightsFile+'_lastEpoch.h5')
    print("Saved weights to",weightsDir+weightsFile)
    utils.plot_history(history, plotDir,['loss','accuracy'])

    # save the train and validation batches
    np.save('e_'+trainFile, trainBatchesE)
    np.save('bkg_'+trainFile, trainBatchesBkg)
    np.save('e_'+valFile, valBatchesE)
    np.save('bkg_'+valFile, valBatchesBkg)

  if(validate):

    # load the batches used to train and validate
    valBatchesE = np.load('e_'+valFile+'.npy')
    valBatchesBkg = np.load('bkg_'+valFile+'.npy')

    validatedE, validatedBkg = 0,0
    for iBatch,batch in enumerate(valBatchesE):
      for iFile,file in enumerate(batch):
        fname = "e_"+tag+file+".npy"
        temp = np.load(dataDir+fname) 
        images = temp[:,1:]
        images = np.reshape(images, [len(images),40,40,4])
        if(iFile == 0): x_test = images[:,:,:,[0,2,3]]
        else: x_test = np.concatenate((x_test,images[:,:,:,[0,2,3]]))
      
      x_test = x_test[:nElectronsPerBatch]
      validatedE+=x_test.shape[0]
      if(iBatch==0): predictionsE = model.predict(x_test)
      else: predictionsE = np.concatenate([predictionsE, model.predict(x_test)])

    for iBatch,batch in enumerate(valBatchesBkg):
      for iFile,file in enumerate(batch):
        fname = "bkg_"+tag+file+".npy"
        temp = np.load(dataDir+fname) 
        images = temp[:,1:]
        images = np.reshape(images, [len(images),40,40,4])
        x_test = images[:,:,:,[0,2,3]]
        if(iFile == 0): x_test = images[:,:,:,[0,2,3]]
        else: x_test = np.concatenate((x_test,images[:,:,:,[0,2,3]]))
      
      x_test = x_test[:nBkgPerBatchData]
      validatedBkg+=x_test.shape[0]
      if(iBatch == 0): predictionsB = model.predict(x_test)
      else: predictionsB = np.concatenate([predictionsB, model.predict(x_test)])
          
    predictions = np.concatenate((predictionsE, predictionsB))
    true = np.concatenate((np.ones(len(predictionsE)), np.zeros(len(predictionsB))))
    y_test = keras.utils.to_categorical(true, num_classes=2)

    utils.metrics(y_test, true, predictions, plotDir)

    print()
    print("Validated on",validatedE,"electron events and",validatedBkg,"background events")
    print("Saved metrics to",plotDir)
    print()