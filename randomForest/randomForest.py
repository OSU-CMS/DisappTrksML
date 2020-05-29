#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

#workDir = '/home/mcarrigan/disTracksML/'

#dataDir = '/home/MilliQan/data/disappearingTracks/tracks/'
dataDir = '/Users/michaelcarrigan/Desktop/DisTracks/'
workDir = '/Users/michaelcarrigan/Desktop/DisTracks/'
saveDir = workDir + 'randomForest/'

#import data
data_e = np.load(dataDir + 'e_DYJets50_norm_20x20.npy')
data_bkg = np.load(dataDir + 'bkg_DYJets50_norm_20x20.npy')
classes = np.concatenate([np.ones(len(data_e)), np.zeros(len(data_bkg))])
data = np.concatenate([data_e, data_bkg])

#shuffle data
indicies = np.arange(data.shape[0])
np.random.shuffle(indicies)
data = data[indicies]
classes = classes[indicies]

print(np.shape(data))

ecal = []
hcal = []
muons = []

data_mod = [ecal, hcal, muons]

entries = len(data)

for matrix in range(entries):
    this_matrix = data[matrix]
    this_matrix = this_matrix.reshape((400,3))
    data_ecal = this_matrix[:, 0]
    data_hcal = this_matrix[:, 1]
    data_muon = this_matrix[:, 2]
    #data_ecal.reshape((-1))
    #data_hcal.reshape((-1))
    #data_muon.reshape((-1))
    ecal.append(data_ecal)
    hcal.append(data_hcal)
    muons.append(data_muon)


#print(np.shape(data_mod))
print(np.shape(ecal))
print(np.shape(muons))


x_train, x_test, y_train, y_test = train_test_split(ecal, classes, test_size = 0.2, random_state=0)
model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

print("confusion matrix: " + str(cm))

print("model score: " + str(model.score(x_test, y_test)))

