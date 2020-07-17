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
import pickle

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
    train = True
    validate = True
    trainFile = "trainBatches"
    valFile = "valBatches"

    nTotE = 12500
    val_size = 0.2
    undersample_bkg = -1          
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

    # batches per epoch, number of electrons/bkg events per batch
    nBatches = int((nTotE + nTotBkg)*1.0/batch_size)
    nElectronsPerBatch = int(np.ceil(nTotE/nBatches))
    nBkgPerBatch = int(batch_size - nElectronsPerBatch)
    nBkgPerBatchData = int(nElectronsPerBatch * fBkg / fE)

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

    def chunks(lst, chunk_sizes):
        out = []
        i = 0
        for chunk in chunk_sizes:
            out.append(lst[i:i+chunk])
            i=i+chunk
        return out

    # fill lists of all events and files
    events, files = [], []
    for file, nEvents in bkgCounts.items():
        for evt in range(nEvents):
            events.append(evt)
            files.append(file)
    
    # divide the lists of events and files in chunks as per
    # the number of bkg per batch
    event_batches_full = chunks(events,bkgPerBatch)
    file_batches_full = chunks(files,bkgPerBatch)
    
    bkg_event_batches, bkg_file_batches = [],[]
    batches = 0
    for events, files in zip(event_batches_full, file_batches_full):
        if(batches == nBatches): break
        events = list(map(int, events)) 
        files = list(map(int, files)) 
        files.sort()
        bkg_event_batches.append([events[0],events[-1]])
        bkg_file_batches.append(list(set(files)))
        batches+=1

    events, files = [], []
    for file, nEvents in eCounts.items():
        for evt in range(nEvents):
            events.append(evt)
            files.append(file)

    event_batches_full = chunks(events,ePerBatch)
    file_batches_full = chunks(files,ePerBatch)

    e_event_batches, e_file_batches = [],[]
    batches = 0
    for events, files in zip(event_batches_full, file_batches_full):
        if(batches == nBatches): break
        events = list(map(int, events)) 
        files = list(map(int, files))
        files.sort()
        e_event_batches.append([events[0],events[-1]])
        e_file_batches.append(list(set(files)))
        batches+=1
    
    e_event_batches = np.array(e_event_batches)
    e_file_batches = np.array(e_file_batches)
    bkg_event_batches = np.array(bkg_event_batches)
    bkg_file_batches = np.array(bkg_file_batches)

    bkg_event_batches = bkg_event_batches[:len(e_event_batches)]
    bkg_file_batches = bkg_file_batches[:len(e_event_batches)]
    
    indices = np.arange(len(e_event_batches)).astype(int)
    random.shuffle(indices)
    e_event_batches = e_event_batches[indices]
    e_file_batches = e_file_batches[indices]
    random.shuffle(indices)
    bkg_event_batches = bkg_event_batches[indices]
    bkg_file_batches = bkg_file_batches[indices]

    nTrainBatches = int(len(e_event_batches)*(1-val_size))

    train_e_event_batches = e_event_batches[:nTrainBatches]
    val_e_event_batches = e_event_batches[nTrainBatches:]
    train_e_file_batches = e_file_batches[:nTrainBatches]
    val_e_file_batches = e_file_batches[nTrainBatches:]

    train_bkg_event_batches = bkg_event_batches[:nTrainBatches]
    val_bkg_event_batches = bkg_event_batches[nTrainBatches:]
    train_bkg_file_batches = bkg_file_batches[:nTrainBatches]
    val_bkg_file_batches = bkg_file_batches[nTrainBatches:]

    def count_events(file_batches, event_batches, dict):
        nSaved=0
        for files, indices in zip(file_batches, event_batches):
            lastFile = len(files)-1
            for iFile, file in enumerate(files):
                if(iFile == 0 and iFile != lastFile):
                    nSaved+=(dict[str(file)]-indices[0])

                elif(iFile == lastFile and iFile != 0):
                    nSaved+=(indices[1]+1)

                elif(iFile == 0 and iFile == lastFile):
                    nSaved+=(indices[1]-indices[0]+1)

                elif(iFile != 0 and iFile != lastFile):
                    nSaved+=dict[str(file)]
        return nSaved

    nSavedETrain = count_events(train_e_file_batches, train_e_event_batches, eCounts)
    nSavedEVal = count_events(val_e_file_batches, val_e_event_batches, eCounts)
    nSavedBkgTrain = count_events(train_bkg_file_batches, train_bkg_event_batches, bkgCounts)
    nSavedBkgVal = count_events(val_bkg_file_batches, val_bkg_event_batches, bkgCounts)

    print("\t\tElectrons\tBackground\te/(e+bkg)")
    print("Requested:\t"+str(nTotE)+"\t\t"+str(nTotBkg)+"\t\t"+str(round(nTotE*1.0/(nTotE+nTotBkg),5)))
    print("Training on:\t"+str(nSavedETrain)+"\t\t"+str(nSavedBkgTrain)+"\t\t"+str(round(nSavedETrain*1.0/(nSavedETrain+nSavedBkgTrain),5)))
    print("Validating on:\t"+str(nSavedEVal)+"\t\t"+str(nSavedBkgVal)+"\t\t"+str(round(nSavedEVal*1.0/(nSavedEVal+nSavedBkgVal),5)))
    print("Dataset:\t"+str(availableE)+"\t\t"+str(availableBkg)+"\t\t"+str(round(fE,5)))

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
    # if oversampling in training data, set appropriate oversample_e in each batch
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

    if(train and class_weights):

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

    elif(train):
        history = model.fit(train_generator, 
                            epochs = epochs,
                            verbose= v,
                            validation_data=val_generator,
                            callbacks=callbacks)
                            
    model.load_weights(weightsDir+weightsFile+'.h5')

    if(train):
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

        

    if(validate):

        # load the batches used to train and validate
        val_e_file_batches = np.load(outputDir+'e_files_'+valFile+'.npy')
        val_e_event_batches = np.load(outputDir+'e_events_'+valFile+'.npy')
        val_bkg_file_batches = np.load(outputDir+'bkg_files_'+valFile+'.npy')
        val_bkg_event_batches = np.load(outputDir+'bkg_events_'+valFile+'.npy')

        validatedE, validatedBkg = 0,0
        iBatch=0
        for files, indices in zip(val_e_file_batches, val_e_event_batches):
            lastFile = len(files)-1
            files.sort()
            for iFile, file in enumerate(files):
                if(iFile == 0 and iFile != lastFile):
                    e_images = np.load(dataDir+'e_'+tag+str(file)+'.npy')[indices[0]:]

                elif(iFile == lastFile and iFile != 0):
                    e_images = np.concatenate((e_images,np.load(dataDir+'e_'+tag+str(file)+'.npy')[:indices[1]+1]))

                elif(iFile == 0 and iFile == lastFile):
                    e_images = np.load(dataDir+'e_'+tag+str(file)+'.npy')[indices[0]:indices[1]+1]

                elif(iFile != 0 and iFile != lastFile):
                    e_images = np.concatenate((e_images,np.load(dataDir+'e_'+tag+str(file)+'.npy')))
        
            e_images = np.reshape(e_images[:,1:],(e_images.shape[0],40,40,4))
            e_images = e_images[:,:,:,[0,2,3]]
            validatedE+=e_images.shape[0]
            if(iBatch==0): predictionsE = model.predict(e_images)
            else: predictionsE = np.concatenate([predictionsE, model.predict(e_images)])
            iBatch+=1

        iBatch=0
        for files, indices in zip(val_bkg_file_batches, val_bkg_event_batches):
            lastFile = len(files)-1
            files.sort()
            for iFile, file in enumerate(files):
                if(iFile == 0 and iFile != lastFile):
                    bkg_images = np.load(dataDir+'bkg_'+tag+str(file)+'.npy')[indices[0]:,:]

                elif(iFile == lastFile and iFile != 0):
                    bkg_images = np.concatenate((bkg_images,np.load(dataDir+'bkg_'+tag+str(file)+'.npy')[:indices[1]+1,:]))

                elif(iFile == 0 and iFile == lastFile):
                    bkg_images = np.load(dataDir+'bkg_'+tag+str(file)+'.npy')[indices[0]:indices[1]+1,:]

                elif(iFile != 0 and iFile != lastFile):
                    bkg_images = np.concatenate((bkg_images,np.load(dataDir+'bkg_'+tag+str(file)+'.npy')))
            
            bkg_images = np.reshape(bkg_images[:,1:],(bkg_images.shape[0],40,40,4))
            bkg_images = bkg_images[:,:,:,[0,2,3]]
            validatedBkg+=bkg_images.shape[0]
            if(iBatch==0): predictionsB = model.predict(bkg_images)
            else: predictionsB = np.concatenate([predictionsB, model.predict(bkg_images)])
            iBatch+=1
        
                    
        predictions = np.concatenate((predictionsE, predictionsB))
        true = np.concatenate((np.ones(len(predictionsE)), np.zeros(len(predictionsB))))
        #y_test = keras.utils.to_categorical(true, num_classes=2)

        utils.metrics(true, predictions, plotDir, threshold=0.5)

        print()
        print(utils.bcolors.HEADER+"Validated on "+str(validatedE)+" electron events and "+str(validatedBkg)+" background events"+utils.bcolors.ENDC)
        print(utils.bcolors.GREEN+"Saved metrics to "+plotDir+utils.bcolors.ENDC)
        print()