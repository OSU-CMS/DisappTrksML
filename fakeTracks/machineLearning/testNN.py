from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import os
import numpy as np
from sklearn.model_selection import train_test_split


def loadData(dataDir = "/store/user/mcarrigan/fakeTracks/converted_v1/"):
    #layerId, charge, isPixel, pixelHitSize, pixelHitSizeX, pixelHitSizeY, stripShapeSelection, hitPosX, hitPosY
    # load the dataset
    file_count = 0
    for filename in os.listdir(dataDir):
        print("Loading...", dataDir + filename)
        if file_count > 10: break
        myfile = np.load(dataDir+filename)
        fakes = np.array(myfile["fake_infos"])
        reals = np.array(myfile["real_infos"])
        if(file_count == 0):
            fakeTracks = fakes
            realTracks = reals
        else:
            fakeTracks = np.concatenate((fakeTracks, fakes))
            realTracks = np.concatenate((realTracks, reals))
        file_count += 1


    print("Number of fake tracks:", len(fakeTracks))
    print("Number of real tracks:", len(realTracks))

    #combine all data and shuffle
    allTracks = np.concatenate((fakeTracks, realTracks))

    indices = np.arange(len(allTracks))
    np.random.shuffle(indices)

    allTracks = allTracks[indices]
    allTracks = np.reshape(allTracks, (-1,156))
    allTracks = np.tanh(allTracks)
    allTruth = np.concatenate((np.ones(len(fakeTracks)), np.zeros(len(realTracks))))
    allTruth = allTruth[indices]

    #split data into train and test

    trainTracks, testTracks, trainTruth, testTruth = train_test_split(allTracks, allTruth, test_size = 0.3)
    return trainTracks, testTracks, trainTruth, testTruth

x, testTracks, y, testTruth = loadData() 

def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=156, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=5, batch_size=5, verbose=1)
history = estimator.fit(x, y)




