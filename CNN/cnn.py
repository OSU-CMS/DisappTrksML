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
    model.add(keras.layers.Dense(2, activation='softmax',bias_initializer=keras.initializers.Constant(0)))
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])
    #print(model.summary())

    return model


def train_model(model, x_train, y_train, x_test, y_test, weightsDir, weightsFile, patience_count = 20, epochs = 100, batch_size = 128, class_weights=True):

  neg, pos = np.bincount(y_train)

  if(class_weights):
    weight_for_0 = (1/neg)*(neg+pos)/2.0
    weight_for_1 = (1/pos)*(neg+pos)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}

  y_train = keras.utils.to_categorical(y_train, 2)
  y_test = keras.utils.to_categorical(y_test, 2)

  callbacks = [
    keras.callbacks.EarlyStopping(patience=patience_count),
    keras.callbacks.ModelCheckpoint(filepath=weightsDir+weightsFile+'.h5',
                                    save_weights_only=True,
                                    monitor='val_loss',
                                    mode='auto',
                                    save_best_only=True),
  ]

  if(class_weights):
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=callbacks,
              class_weight = class_weight)
  else:
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test),
              callbacks=callbacks)

  return history


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
  pos_class = [1]
  neg_class = [0,2]
  batch_size = 256
  epochs = 10
  patience_count = 10
  img_rows, img_cols = 40, 40
  channels = 3
  input_shape = (img_rows,img_cols,channels)
  class_weights = True
  oversample_val = 0.1
  undersample_val = 0.3


  os.system('mkdir '+str(plotDir))
  os.system('mkdir '+str(weightsDir))


  """
  infos:

  0: ID
  1: matched track gen truth flavor (1: electrons, 2: muons, 0: everything else)
  2: nPV
  3: deltaRToClosestElectron
  4: deltaRToClosestMuon
  5: deltaRToClosestTauHaud
  6: classes 1: electron, 0: background (electron selection only)

  """

  images, infos = utils.load_electron_data(dataDir, tag)

  x = images[:,:-1]
  x = np.reshape(x, [len(x),40,40,4])
  x = x[:,:,:,[0,2,3]]

  y = np.array([x[6] for x in infos])
  y = y.astype(int)

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')

  # x_train, y_train = utils.apply_oversampling(x_train,y_train,oversample_val=oversample_val)
  # x_train, y_train = utils.apply_undersampling(x_train,y_train,undersample_val=undersample_val)

  # initialize output bias
  neg, pos = np.bincount(y_train)
  output_bias = np.log(pos/neg)
  output_bias = keras.initializers.Constant(output_bias)
  print("Positive Class Counter:",pos)
  print("Negative Class Counter:",neg)

  model = build_model(input_shape = input_shape, 
                      layers = 5, filters = 64, opt='adam',
                      output_bias=output_bias)

  weightsFile = 'first_model'

  history = train_model(model,x_train,y_train,x_test,y_test,
                        weightsDir,weightsFile,
                        patience_count=patience_count,
                        epochs=epochs,
                        batch_size=batch_size,
                        class_weights=class_weights)

  utils.plot_history(history, plotDir)

  model.load_weights(weightsDir+weightsFile+'.h5')
  predictions = model.predict(x_test)

  print()
  print("Calculating and plotting confusion matrix")
  cm = utils.calc_cm(y_test,predictions)
  plot_confusion_matrix(cm,['bkg','e'],plotDir + 'cm.png')
  print()

  print("Plotting ceratainty")
  utils.plot_certainty(y_test,predictions,plotDir+'certainty.png')
  print()

  precision, recall = utils.calc_binary_metrics(cm)
  print("Precision = TP/(TP+FP) = fraction of predicted true actually true ",round(precision,3))
  print("Recall = TP/(TP+FN) = fraction of true class predicted to be true ",round(recall),3)
  auc = roc_auc_score(y_test,predictions)
  print("AUC Score:",round(auc,5))
  print()