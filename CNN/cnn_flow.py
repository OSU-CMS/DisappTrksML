import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.pipeline import Pipeline
import utils
import json
import random


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

  

class generator(keras.utils.Sequence):
  
  def __init__(self, filenames, batch_size):
    self.filenames = filenames
    self.batch_size = batch_size
    
  def __len__(self) :
    return (np.ceil(len(self.filenames*self.batch_size) / float(self.batch_size))).astype(np.int)

  def __getitem__(self, idx) :

    filename = self.filenames[idx]
    file = np.load(dataDir+'images'+tag+'_'+str(filename)+'.npz')

    images = file['images']
    infos = file['infos']

    batch_indices = list(range(0,images.shape[0]))
    random.shuffle(batch_indices)

    images = images[batch_indices[:batch_size],1:]
    batch_x = np.reshape(images,(batch_size,40,40,4))
    batch_x = batch_x[:,:,:,[0,2,3]]

    batch_y = infos[batch_indices[:batch_size],1]
    for i in range(len(batch_y)): 
      if int(batch_y[i]) != 1: batch_y[i] = 0
      else: batch_y[i] = 1
    batch_y = keras.utils.to_categorical(batch_y, 2)

    return np.array(batch_x), np.array(batch_y)


if __name__ == "__main__":

  # limit CPU usage
  config = tf.ConfigProto(inter_op_parallelism_threads = 2,   
                          intra_op_parallelism_threads = 2)
  tf.keras.backend.set_session(tf.Session(config=config))

  dataDir = '/store/user/llavezzo/electron_selection/'
  tag = '_0p25_tanh'
  workDir = '/data/users/llavezzo/cnn/'
  plotDir = workDir + 'plots/'
  weightsDir = workDir + 'weights/'

  #config parameters
  pos,neg = 431, 91805              # from count.py
  batch_size = 256
  epochs = 10
  patience_count = 10
  img_rows, img_cols = 40, 40
  channels = 3
  input_shape = (img_rows,img_cols,channels)
  class_weights = True
  # oversample_val = 0.1
  # undersample_val = 0.3


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


  with open(dataDir+'eventCounts.json') as json_file:
    eventCounts = json.load(json_file)

  files = []
  nTotalEvents = 0
  for key, value in eventCounts.items():
    nSamples = int(np.ceil(value/batch_size))
    for _ in range(nSamples): files.append(key) 
    nTotalEvents += value

  trainFiles, valFiles = [],[]
  trainCount, valCount = 0,0
  val_size = 0.2
  uniqueFiles = list(eventCounts.keys())
  random.shuffle(uniqueFiles)
  for file in uniqueFiles:
    if(trainCount < (1-val_size)*nTotalEvents):
      trainFiles.append(file)
      trainCount += eventCounts[file]
    else:
      valFiles.append(file)
      valCount += eventCounts[file]
  

  train_generator = generator(trainFiles, batch_size)
  val_generator = generator(valFiles, batch_size)

  # initialize output bias
  output_bias = np.log(pos/neg)

  model = build_model(input_shape = input_shape, 
                      layers = 5, filters = 64, opt='adam',
                      output_bias=output_bias)

  weightsFile = 'first_model'

  weight_for_0 = (1/neg)*(neg+pos)/2.0
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

  # history = model.fit_generator(train_generator, 
  #                               epochs = 10,
  #                               verbose= 1,
  #                               validation_data=val_generator,
  #                               callbacks=callbacks,
  #                               class_weight=class_weights)

  #utils.plot_history(history, plotDir,['loss','acc'])

  model.load_weights(weightsDir+weightsFile+'.h5')


  full,infos = [],[]
  for file in valFiles:
    fname = 'images'+tag+'_'+str(file)+'.npz'
    temp = np.load(dataDir+fname)
    full.append(temp['images'])
    infos.append(temp['infos'])
  full = np.vstack(full)
  infos = np.vstack(infos)

  images = full[:,1:]
  images = np.reshape(images, [len(images),40,40,4])
  x_test = images[:,:,:,[0,2,3]]
  y_test = infos[:,1]
  for i in range(len(y_test)):
    if(int(y_test[i]) == 1): y_test[i] = 1
    else: y_test[i] = 0 

  predictions = model.predict(x_test)

  y_test = keras.utils.to_categorical(y_test, num_classes=2)

  print()
  print("Calculating and plotting confusion matrix")
  cm = utils.calc_cm(y_test,predictions)
  utils.plot_confusion_matrix(cm,['bkg','e'],plotDir + 'cm.png')
  print()

  print("Plotting ceratainty")
  #utils.plot_certainty(y_test,predictions,plotDir+'certainty.png')
  print()

  precision, recall = utils.calc_binary_metrics(cm)
  print("Precision = TP/(TP+FP) = fraction of predicted true actually true ",round(precision,3))
  print("Recall = TP/(TP+FN) = fraction of true class predicted to be true ",round(recall,3))
  auc = roc_auc_score(y_test,predictions)
  print("AUC Score:",round(auc,5))
  print()

  fileOut = open(plotDir+"metrics.txt","a")
  fileOut.write("Precision = TP/(TP+FP) = fraction of predicted true actually true "+str(round(precision,3)))
  fileOut.write("Recall = TP/(TP+FN) = fraction of true class predicted to be true "+str(round(recall,3)))
  fileOut.write("AUC Score:"+str(round(auc,5)))
  fileOut.close()