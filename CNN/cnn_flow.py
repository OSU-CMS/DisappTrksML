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
  
  def __init__(self, filenames, batch_size, dataDir, e_fraction=-1):
    self.filenames = filenames
    self.batch_size = batch_size
    self.dataDir = dataDir
    self.e_fraction = e_fraction
    
  def __len__(self) :
    return (np.ceil(len(self.filenames*self.batch_size) / float(self.batch_size))).astype(np.int)

  def __getitem__(self, idx) :

    filename = self.filenames[idx]

    # import images
    file_bkg = np.load(self.dataDir+'bkg_'+tag2+str(filename)+'.npz')
    file_e = np.load(self.dataDir+'e_'+tag1+str(filename)+'.npz')
    images_bkg = file_bkg['images']
    images_e = file_e['images']

    # store number of images
    numE = images_e.shape[0]
    numBkg = images_bkg.shape[0]

    # fill the batch with e_fraction electron images,
    # oversampling if needed (none by default)
    images_e = images_e[:,1:]
    added_e = []
    while((numE+len(added_e))*1.0 / batch_size < self.e_fraction):
      added_e.append(images_e[random.randint(0,numE-1),:])
    
    # number of added electrons
    numAdded = len(added_e)
    if(numAdded > 0): added_e = np.reshape(added_e,[numAdded, 6400])

    # fill the rest of the batch with random bkg images
    batch_indices = list(range(0,images_bkg.shape[0]))
    random.shuffle(batch_indices)
    images_bkg = images_bkg[batch_indices[(numE+numAdded):batch_size],1:]
    
    # join and reshape the images
    if(numAdded > 0): batch_x = np.concatenate((images_e,added_e,images_bkg))
    else: batch_x = np.concatenate((images_e,images_bkg))
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
  config = tf.ConfigProto(inter_op_parallelism_threads = 2,   
                          intra_op_parallelism_threads = 2)
  tf.keras.backend.set_session(tf.Session(config=config))

  dataDir = "/store/user/llavezzo/disappearingTracks/electron_selection_DYJetsToll_M50/"
  tag1 = '0p25_tanh_'
  tag2 = '0p25_tahn_'
  workDir = '/data/users/llavezzo/cnn/'
  plotDir = workDir + 'plots/'
  weightsDir = workDir + 'weights/'

  ################config parameters################
  pos,neg = 431, 91805              # from splt.py
  batch_size = 256
  epochs = 10
  patience_count = 10
  maxFiles = 1000
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

  # store each file number proportionally
  # to how many events it has and the batch size
  files = []
  nFiles = 0
  nTotalEvents = 0
  for (file, numE),(file,numBkg) in zip(eCounts.items(),bkgCounts.items()):
    nSamples = int(np.ceil((numE+numBkg)/batch_size))   #FIXME: should we sample more from each file?
    for _ in range(nSamples): files.append(file) 
    nTotalEvents += (numE+numBkg)
    nFiles += 1
    if(nFiles >= maxFiles): break

  # split files into train and validation sets
  trainFiles, valFiles = [],[]
  trainCount, valCount = 0,0
  uniqueFiles = list(eCounts.keys())
  uniqueFiles = uniqueFiles[:nFiles]
  random.shuffle(uniqueFiles)
  for file in uniqueFiles:
    if(trainCount < (1-val_size)*nTotalEvents):
      trainFiles.append(file)
      trainCount += (eCounts[file]+bkgCounts[file])
    else:
      valFiles.append(file)
      valCount += (eCounts[file]+bkgCounts[file])


  # initialize generators
  train_generator = generator(trainFiles, batch_size, dataDir, e_fraction)
  val_generator = generator(valFiles, batch_size, dataDir, e_fraction)

  # initialize output bias
  output_bias = np.log(pos/neg)       #FIXME: doesn't work

  model = build_model(input_shape = input_shape, 
                      layers = 5, filters = 64, opt='adam',
                      output_bias=0)

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

  history = model.fit_generator(train_generator, 
                                epochs = 10,
                                verbose= 1,
                                validation_data=val_generator,
                                callbacks=callbacks)                                #class_weight=class_weights)

  model.load_weights(weightsDir+weightsFile+'.h5')

  # predict each of the validation files individually
  predictions, true = [],[]
  for file in valFiles:
    f = tag+'_'+str(file)+'.npz'
    temp1 = np.load(dataDir+"e_"+f)
    temp2 = np.load(dataDir+"bkg_"+f) 
    images_e = temp1['images'][:,1:]
    images_bkg = temp2['images'][:,1:]
    images = np.concatenate((images_e,images_bkg))
    images = np.reshape(images, [len(images),40,40,4])
    x_test = images[:,:,:,[0,2,3]]
    y_test = np.concatenate((np.ones(len(images_e)),np.zeros(len(imges_bkg))))
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