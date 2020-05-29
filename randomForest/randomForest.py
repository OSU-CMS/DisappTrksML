#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import sklearn
import sys


if(len(sys.argv) < 2): iterations = 10
else: iterations = int(sys.argv[1])

if(len(sys.argv) < 3): training_pct = .20
else: training_pct = float(sys.argv[2])

def precision(cm):
    if(cm[1][1]==0 and cm[1][0]): p = 0
    else: p = float(cm[1][1])/(float(cm[1][1]) + float(cm[1][0]))
    return p

def recall(cm):
    if(cm[1][1]==0 and cm[0][1]==0): r = 0
    else: r = float(cm[1][1])/(float(cm[1][1]) + float(cm[0][1]))
    return r

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
def shuffle(data, classes):
    indicies = np.arange(data.shape[0])
    np.random.shuffle(indicies)
    data = data[indicies]
    classes = classes[indicies]

    #print(np.shape(data))

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
        ecal.append(data_ecal)
        hcal.append(data_hcal)
        muons.append(data_muon)

        #print(np.shape(ecal))
        #print(np.shape(muons))
    return data_mod

e_scores = []
h_scores = []
m_scores = []
scores = [e_scores, h_scores, m_scores]

names = ["ecal", "hcal", "muon"]

for i in range(3):
    score_avg = 0
    r_avg = 0
    p_avg = 0
    for counter in range(iterations):
        
        data_mod = shuffle(data, classes)
        #print(np.shape(data_mod[i]))
        #print(data_mod[i][:5])
        x_train, x_test, y_train, y_test = train_test_split(data_mod[i], classes, test_size = training_pct, random_state=0)
        model = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        this_cm = np.array(confusion_matrix(y_test, y_pred))
        #tn, fn, fp, tp
        
        score = model.score(x_test, y_test)
        scores[i].append(score)
        score_avg += score
        r_avg += recall(this_cm)
        p_avg += precision(this_cm)
        #print(this_cm) 
        #print(str(precision(this_cm)) + " " + str(recall(this_cm)))
        #print("model score: " + str(score))
        #print("f1: " + str(sklearn.metrics.f1_score(y_test, y_pred)))
        #print("precision: " + str(sklearn.metrics.precision_score(y_test, y_pred)))
        if(counter == iterations-1):
            score_avg = score_avg/iterations
            r_avg = r_avg/iterations
            p_avg = p_avg/iterations
            print("average " + names[i] + " model score: " + str(score_avg) + " in " + str(iterations) + " trials \n"
                  "average " + names[i] + " precision: " + str(p_avg) + "\n"
                  "average " + names[i] + " recall: " + str(r_avg))
   
    

