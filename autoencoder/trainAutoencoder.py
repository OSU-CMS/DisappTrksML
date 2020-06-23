import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras import optimizers
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import scipy
import keras
import sys

file_num = str(sys.argv[1])
if(len(sys.argv) > 3): loadWeights = str(sys.argv[3])
saveWeights = str(sys.argv[2])


#set epochs, batch size
num_epochs = 30
batches = 16

#set directories
workDir = "/home/mcarrigan/disTracksML/autoencoder/"
saveDir = "/home/mcarrigan/disTracksML/weights/autoencoder/"
dataDir = "/home/mcarrigan/disTracksML/data/v2/AE/"

#load data
df = pd.read_pickle(dataDir+'images_0p25_' + file_num + '.pkl')
data = df.iloc[:,4:].to_numpy()
data = np.reshape(data, (-1, 40, 40, 4))
df_e = df.loc[df['type'] == 1]
df_bkg = df.loc[df['type'] ==0]
df_m = df.loc[df['type'] == 2]
data_e = df_e.iloc[:, 4:].to_numpy()
data_e = np.reshape(data_e, (data_e.shape[0], 40, 40, 4))
data_e = data_e[:, :, :, [0,2,3]]
data_bkg = df_bkg.iloc[:, 4:].to_numpy()
data_bkg = np.reshape(data_bkg, (data_bkg.shape[0],40,40,4))
data_bkg = data_bkg[:, :, :, [0,2,3]]
data_m = df_m.iloc[:, 4:].to_numpy()
data_m = np.reshape(data_m, (-1, 40, 40, 4))
data_m = data_m[:, :, :, [0,2,3]]
classes = np.array([np.ones(len(data_e)), np.ones(len(data_e)), np.ones(len(data_e))])
classes = np.reshape(classes, (len(data_e), 3))

data = data_e

#shuffle data
indicies = np.arange(data.shape[0])
np.random.shuffle(indicies)
data = data[indicies]
print(data.shape[0], "Number of samples")
print(np.shape(data))
print(np.shape(classes))

x_train, x_val, y_train, y_val = train_test_split(data, classes, test_size=0.15, random_state=0)

print("Training Samples", x_train.shape[0])
print("Validation Samples", x_val.shape[0])

x_train = x_train[:, :, :, 0]
x_val = x_val[:, :, :, 0]

fig, plot = plt.subplots(1,5, figsize = (10,5))
for i in range(5):
    rand = np.random.randint(0, x_train.shape[0])
    plot[i].imshow(x_train[rand, :, :])

plt.savefig(workDir + 'TrainElectrons.png')

x_train = np.reshape(x_train, (-1, 40, 40, 1))
x_val = np.reshape(x_val, (-1, 40, 40, 1))

def noisyData(data_e, data_bkg):
    evts = len(data_e)
    noise = np.random.randint(0, len(data_bkg), evts)
    data_bkg = data_bkg[noise]
    data_noise = np.copy(data_e)
    for i in range(len(data_e)):
        index = np.random.randint(0, len(data_bkg))
        this_bkg = data_bkg[index, :, :, 0]
        norm_bkgE = np.linalg.norm(this_bkg)
        norm_E = np.linalg.norm(data_e[i, :, :, 0])
        while(norm_bkgE < norm_E*0.5): 
            index2 = np.random.randint(0, len(data_bkg))
            this_bkg = np.add(this_bkg, data_bkg[index2, :, :, 0])
            norm_bkgE = np.linalg.norm(this_bkg)
        data_noise[i, :, :, 0] = np.add(data_e[i, :, :, 0], data_bkg[i, :, :, 0])
    return data_noise

noisyTrain = noisyData(x_train, data_bkg)
noisyVal = noisyData(x_val, data_bkg)

img = Input(shape = (40,40,1))
ae = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(img)
#ae = MaxPooling2D((2,2), padding = 'same')(ae)
#encoded = Conv2D(32, (3,3), strides = (2,2), activation = 'relu', padding = 'same')(ae)
#ae = Conv2D(32, (3,3), activation = 'relu', padding = "same")(ae)
encoded = MaxPooling2D((2,2), padding = 'same')(ae)
#encoded = Conv2D(16, (3,3), strides = (2,2), activation = 'relu', padding = 'same')(ae)

ae = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(encoded)
#ae = Conv2DTranspose(32, (3,3), activation = 'relu', strides = (2,2), padding = 'same')(ae)
ae = UpSampling2D((2,2))(ae)
#ae = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(ae)
#ae = UpSampling2D((2,2))(ae)
#ae = Conv2DTranspose(128, (3,3), strides = (2,2), activation = 'relu', padding = 'same')(ae)
decoded = Conv2D(1, (3,3), activation = 'sigmoid', padding = 'same')(ae)


autoencoder = Model(img, decoded)
if(len(sys.argv) > 3): 
    autoencoder.load_weights(saveDir + loadWeights)
    print("loading weights", loadWeights)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(noisyTrain, x_train, epochs = num_epochs, batch_size = batches, shuffle=True, validation_data = (noisyVal, x_val))

autoencoder.save_weights(saveDir+saveWeights)

