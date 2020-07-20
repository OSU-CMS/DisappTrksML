import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
import random
import sys
import pickle
import utils
import validate

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
    model.add(keras.layers.Dense(1, activation='sigmoid',bias_initializer=keras.initializers.Constant(output_bias)))
    model.compile(loss=keras.losses.BinaryCrossentropy(),
              optimizer=opt,
              metrics=metrics)
    #print(model.summary())
    
    return model
  
# generate batches of images from files
class generator(keras.utils.Sequence):
  
  def __init__(self, batchesE, batchesBkg, indicesE, indicesBkg, batch_size, dataDir):
    self.batchesE = batchesE
    self.batchesBkg = batchesBkg
    self.indicesE = indicesE
    self.indicesBkg = indicesBkg
    self.batch_size = batch_size
    self.dataDir = dataDir
    
  def __len__(self) :
    return len(self.batchesE)

  def __getitem__(self, idx) :

    filenamesE = self.batchesE[idx]
    filenamesBkg = self.batchesBkg[idx]
    indexE = self.indicesE[idx]
    indexBkg = self.indicesBkg[idx]

    lastFile = len(filenamesE)-1
    filenamesE.sort()
    for iFile, file in enumerate(filenamesE):
        if(iFile == 0 and iFile != lastFile):
            e_images = np.load(self.dataDir+'e_'+tag+str(file)+'.npy')[indexE[0]:]

        elif(iFile == lastFile and iFile != 0):
            e_images = np.concatenate((e_images,np.load(self.dataDir+'e_'+tag+str(file)+'.npy')[:indexE[1]+1]))

        elif(iFile == 0 and iFile == lastFile):
            e_images = np.load(self.dataDir+'e_'+tag+str(file)+'.npy')[indexE[0]:indexE[1]+1]

        elif(iFile != 0 and iFile != lastFile):
            e_images = np.concatenate((e_images,np.load(self.dataDir+'e_'+tag+str(file)+'.npy')))
    
    
    lastFile = len(filenamesBkg)-1
    filenamesBkg.sort()
    for iFile, file in enumerate(filenamesBkg):
        if(iFile == 0 and iFile != lastFile):
            bkg_images = np.load(self.dataDir+'bkg_'+tag+str(file)+'.npy')[indexBkg[0]:,:]

        elif(iFile == lastFile and iFile != 0):
            bkg_images = np.concatenate((bkg_images,np.load(self.dataDir+'bkg_'+tag+str(file)+'.npy')[:indexBkg[1]+1]))

        elif(iFile == 0 and iFile == lastFile):
            bkg_images = np.load(self.dataDir+'bkg_'+tag+str(file)+'.npy')[indexBkg[0]:indexBkg[1]+1]

        elif(iFile != 0 and iFile != lastFile):
            bkg_images = np.concatenate((bkg_images,np.load(self.dataDir+'bkg_'+tag+str(file)+'.npy')))
    

    numE = e_images.shape[0]
    numBkg = self.batch_size-numE
    bkg_images = bkg_images[:numBkg]

    # shuffle and select appropriate amount of electrons, bkg
    indices = list(range(e_images.shape[0]))
    random.shuffle(indices)
    e_images = e_images[indices,1:]

    indices = list(range(bkg_images.shape[0]))
    bkg_images = bkg_images[indices,1:]

    # concatenate images and suffle them, create labels
    batch_x = np.concatenate((e_images,bkg_images))
    batch_y = np.concatenate((np.ones(numE),np.zeros(numBkg)))
    
    indices = list(range(batch_x.shape[0]))
    random.shuffle(indices)

    batch_x = batch_x[indices[:self.batch_size],:]
    batch_x = np.reshape(batch_x,(batch_size,40,40,4))
    batch_x = batch_x[:,:,:,[0,2,3]]


    batch_y = batch_y[indices[:self.batch_size]]
    #batch_y = keras.utils.to_categorical(batch_y, num_classes=2)

    return batch_x, batch_y


if __name__ == "__main__":

    # limit CPU usage
    # config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads = 2,   
    #                         intra_op_parallelism_threads = 2)
    # tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    # suppress warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


    dataDir = "/data/disappearingTracks/electron_selectionV2/"
    tag = '0p25_tanh_'
    workDir = '/home/llavezzo/'
    plotDir = workDir + 'plots/cnn/'
    weightsDir = workDir + 'weights/cnn/'
    outputDir = workDir + 'outputFiles/cnn/'
    weightsFile = 'cnn'

    ################config parameters################
    """
    train/val Files:
    saves the names of the events/files for train/val data
    for reproducibility

    nTotE:
    how many electron events to use
    from maximum 15463 electron events
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

    run_validate = True
    trainFile = "trainBatches"
    valFile = "valBatches"

    nTotE = 100
    val_size = 0.2
    undersample_bkg = 0.9        
    oversample_e = -1   

    v = 2
    batch_size = 256
    epochs = 1
    patience_count = 5
    monitor = 'val_precision'
    class_weights = True  
    metrics = ['Precision', 'Recall',
            'TruePositives','TrueNegatives',
            'FalsePositives', 'FalseNegatives']

    img_rows, img_cols = 40, 40
    channels = 3
    input_shape = (img_rows,img_cols,channels)
    #################################################


    if(not os.path.isdir(plotDir)): os.system('mkdir '+str(plotDir))
    if(not os.path.isdir(weightsDir)): os.system('mkdir '+str(weightsDir))
    if(not os.path.isdir(outputDir)): os.system('mkdir '+str(outputDir))

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

    # batches per epoch
    nBatches = int((nTotE + nTotBkg)*1.0/batch_size)

    # count how many e/bkg events in each batch
    ePerBatch = np.zeros(nBatches)
    iBatch = 0
    while np.sum(ePerBatch) < nTotE:
        ePerBatch[iBatch]+=1
        iBatch+=1
        if(iBatch == nBatches): iBatch = 0
    bkgPerBatch = np.asarray([batch_size-np.min(ePerBatch)]*nBatches)
    ePerBatch = ePerBatch.astype(int)
    bkgPerBatch = bkgPerBatch.astype(int)

    # fill lists of all events and files
    b_events, b_files = [], []
    for file, nEvents in bkgCounts.items():
        for evt in range(nEvents):
            b_events.append(evt)
            b_files.append(file)
    e_events, e_files = [], []
    for file, nEvents in eCounts.items():
        for evt in range(nEvents):
            e_events.append(evt)
            e_files.append(file)
    
    # make batches
    bkg_event_batches, bkg_file_batches = utils.make_batches(b_events, b_files, bkgPerBatch, nBatches)
    e_event_batches, e_file_batches = utils.make_batches(e_events, e_files, ePerBatch, nBatches)
    
    # train/validation split
    train_e_event_batches, val_e_event_batches, train_e_file_batches, val_e_file_batches = train_test_split(e_event_batches, e_file_batches, test_size=val_size, random_state=42)
    train_bkg_event_batches, val_bkg_event_batches, train_bkg_file_batches, val_bkg_file_batches = train_test_split(bkg_event_batches, bkg_file_batches, test_size=val_size, random_state=42)

    # count events in each batch
    nSavedETrain = utils.count_events(train_e_file_batches, train_e_event_batches, eCounts)
    nSavedEVal = utils.count_events(val_e_file_batches, val_e_event_batches, eCounts)
    nSavedBkgTrain = utils.count_events(train_bkg_file_batches, train_bkg_event_batches, bkgCounts)
    nSavedBkgVal = utils.count_events(val_bkg_file_batches, val_bkg_event_batches, bkgCounts)

    if(nSavedEVal*1.0/(nSavedEVal+nSavedBkgVal) > fE):
        nBkgToLoad = int(nSavedEVal*(1-fE)/fE-nSavedBkgVal)
        lastFile = bkg_file_batches[-1][-1]

        b_events, b_files = [], []
        reached = False
        for file, nEvents in bkgCounts.items():
            if(int(file) != lastFile and not reached): continue
            else: reached = True

            for evt in range(nEvents):
                b_events.append(evt)
                b_files.append(file)
    
        # make batches of 1000 bkg files
        nBatches = int(nBkgToLoad*1.0/1000)
        bkgPerBatch = [1000]*nBatches
               
        bkg_event_batches_added, bkg_file_batches_added = utils.make_batches(b_events, b_files, bkgPerBatch, nBatches)

        nAddedBkg = utils.count_events(bkg_file_batches, bkg_event_batches, bkgCounts)

    val_bkg_event_batches = np.concatenate((val_bkg_event_batches,bkg_event_batches_added))
    val_bkg_file_batches = np.concatenate((val_bkg_file_batches,bkg_file_batches_added))

    nSavedBkgVal = utils.count_events(val_bkg_file_batches, val_bkg_event_batches, bkgCounts)

    print("\t\tElectrons\tBackground\te/(e+bkg)")
    print("Requested:\t"+str(nTotE)+"\t\t"+str(nTotBkg)+"\t\t"+str(round(nTotE*1.0/(nTotE+nTotBkg),5)))
    print("Training on:\t"+str(nSavedETrain)+"\t\t"+str(nSavedBkgTrain)+"\t\t"+str(round(nSavedETrain*1.0/(nSavedETrain+nSavedBkgTrain),5)))
    print("Validating on:\t"+str(nSavedEVal)+"\t\t"+str(nSavedBkgVal)+"\t\t"+str(round(nSavedEVal*1.0/(nSavedEVal+nSavedBkgVal),5)))
    print("Dataset:\t"+str(availableE)+"\t\t"+str(availableBkg)+"\t\t"+str(round(fE,5)))

    sys.exit(0)

    # # oversample the training electron files if oversample_e != -1
    # nElectronsPerBatchOversampled = int(np.ceil(batch_size*oversample_e))
    # ovsFiles = list([file for batch in trainBatchesE for file in batch])
    # random.shuffle(ovsFiles)
    # for i,batch in enumerate(trainBatchesE):
    #     nElectronsThisBatch = 0
    #     for file in batch: nElectronsThisBatch+=eCounts[file]
    #     while nElectronsThisBatch < nElectronsPerBatchOversampled:
    #         randFile = ovsFiles[random.randint(0,len(ovsFiles)-1)]
    #         trainBatchesE[i].append(randFile)
    #         nElectronsThisBatch += eCounts[randFile]

    if(oversample_e != -1):
        print("Oversampling:")
        print("\t Number of electrons per batch:",nElectronsPerBatchOversampled)
        print("\t",len(trainBatchesE),"batches of files (approx.",nElectronsPerBatchOversampled*len(trainBatchesE),"electron and",(batch_size-nElectronsPerBatchOversampled)*len(trainBatchesE), "background events)")

    # initialize generators
    train_generator = generator(train_e_file_batches, train_bkg_file_batches, train_e_event_batches, train_bkg_event_batches, batch_size, dataDir)
    val_generator = generator(val_e_file_batches, val_bkg_file_batches, val_e_event_batches, val_bkg_event_batches, batch_size, dataDir)

    # initialize output bias
    if(oversample_e == -1): output_bias = np.log(nTotE/nTotBkg)
    else: output_bias = np.log(1.0*oversample_e/(1-oversample_e))

    model = build_model(input_shape = input_shape, 
                        layers = 5, filters = 64, opt='adam',
                        output_bias=output_bias,
                        metrics=metrics)

    callbacks = [
    keras.callbacks.EarlyStopping(patience=patience_count),
    keras.callbacks.ModelCheckpoint(filepath=weightsDir+weightsFile+'.h5',
                                    save_weights_only=True,
                                    monitor='val_precision',
                                    mode='auto',
                                    save_best_only=True),
    ]

    if(class_weights):

        # class weights
        if(oversample_e == -1):
            weight_for_0 = (1/nTotBkg)*(nTotBkg+nTotE)/2.0
            weight_for_1 = (1/nTotE)*(nTotBkg+nTotE)/2.0
            class_weight = {0: weight_for_0, 1: weight_for_1}
        else:
            weight_for_0 = 1/(2*(1-oversample_e))
            weight_for_1 = 1/(2.0*oversample_e)
            class_weight = {0: weight_for_0, 1: weight_for_1}

        history = model.fit(train_generator, 
                            epochs = epochs,
                            verbose= v,
                            validation_data=val_generator,
                            callbacks=callbacks,
                            class_weight=class_weight)

    else:
        history = model.fit(train_generator, 
                            epochs = epochs,
                            verbose= v,
                            validation_data=val_generator,
                            callbacks=callbacks)
                            
    model.load_weights(weightsDir+weightsFile+'.h5')

    model.save_weights(weightsDir+weightsFile+'_lastEpoch.h5')
    print(utils.bcolors.GREEN+"Saved weights to "+weightsDir+weightsFile+utils.bcolors.ENDC)

    # save the train and validation batches
    np.save(outputDir+'e_files_'+trainFile, train_e_file_batches)
    np.save(outputDir+'e_events_'+trainFile, train_e_event_batches)
    np.save(outputDir+'e_files_'+valFile, val_e_file_batches)
    np.save(outputDir+'e_events_'+valFile, val_e_event_batches)
    np.save(outputDir+'bkg_files_'+trainFile, train_bkg_file_batches)
    np.save(outputDir+'bkg_events_'+trainFile, train_bkg_event_batches)
    np.save(outputDir+'bkg_files_'+valFile, val_bkg_file_batches)
    np.save(outputDir+'bkg_events_'+valFile, val_bkg_event_batches)
    
    with open(outputDir+'history', 'wb') as f:
        pickle.dump(history.history, f)

    print(utils.bcolors.GREEN+"Saved history, train and validation files to "+outputDir+utils.bcolors.ENDC)

    utils.plot_history(history, plotDir, ['loss','recall','precision'])
    print(utils.bcolors.YELLOW+"Plotted history to "+plotDir+utils.bcolors.ENDC) 

    if(run_validate):
        validate.validate(model, valFile, outputDir, dataDir, tag, plotDir)