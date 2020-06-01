#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import sklearn
import sys


if(len(sys.argv) < 2): iterations = 1
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

workDir = '/home/mcarrigan/disTracksML/'
dataDir = '/home/MilliQan/data/disappearingTracks/tracks/'
#dataDir = '/Users/michaelcarrigan/Desktop/DisTracks/'
#workDir = '/Users/michaelcarrigan/Desktop/DisTracks/'
saveDir = workDir + 'randomForest/'

#import data
data_e = np.load(dataDir + 'e_DYJets50V3_norm_40x40.npy')
data_bkg = np.load(dataDir + 'bkg_DYJets50V3_norm_40x40.npy')
classes = np.concatenate([np.ones(len(data_e)), np.zeros(len(data_bkg))])
data = np.concatenate([data_e, data_bkg])
print("total event count: %i" % len(data))
print("electron count: %i , background count: %i" % (len(data_e), len(data_bkg)))
print(np.shape(data))

#shuffle data


def shuffle(data, classes):
    indicies = np.arange(data.shape[0])
    np.random.shuffle(indicies)
    data_shuffle = data[indicies]
    classes_shuffle = classes[indicies]

    #print(np.shape(data))

    ecal = []
    ecalS = []
    hcal = []
    muons = []

    data_mod = [ecal, ecalS, hcal, muons]

    entries = len(data_shuffle)

    for matrix in range(entries):
        this_matrix = data_shuffle[matrix]
        this_matrix = this_matrix.reshape((1600,5))
        data_ecal = this_matrix[:, 1]
	data_ecalS = this_matrix[:, 2]
        data_hcal = this_matrix[:, 3]
        data_muon = this_matrix[:, 4]
        ecal.append(data_ecal)
	ecalS.append(data_ecalS)
        hcal.append(data_hcal)
        muons.append(data_muon)

        #print(np.shape(ecal))
        #print(np.shape(muons))
    return data_mod, classes_shuffle

e_scores = []
eS_scores = []
h_scores = []
m_scores = []
scores = [e_scores, eS_scores, h_scores, m_scores]

names = ["ecal", "ecal_shower", "hcal", "muon"]

ecalP = []
ecalR = []
ecalSP = []
ecalSR = []
hcalP = []
hcalR = []
muonP = []
muonR = []
precision = [ecalP, ecalSP, hcalP, muonP]
recall = [ecalR, ecalSR, hcalR, muonR]

def runForest(trees):
    precision = []
    recall = []
    for i in range(4):
        score_avg = 0
        r_avg = 0
        p_avg = 0
        for counter in range(iterations):
        
            data_mod, classes_mod = shuffle(data, classes)
            #print(np.shape(data_mod[i]))
            #print(data_mod[i][:5])
            x_train, x_test, y_train, y_test = train_test_split(data_mod[i], classes_mod, test_size = training_pct, random_state=0)
            e_test = 0
            for j in range(len(y_test)):
                if(y_test[j] == 1): e_test += 1
            print("number of electrons in test group: %i" % e_test)
            model = RandomForestClassifier(n_estimators=trees, criterion='entropy', random_state=0)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)

            this_cm = np.array(confusion_matrix(y_test, y_pred))
            score = model.score(x_test, y_test)
            scores[i].append(score)
            score_avg += score
            r_avg += sklearn.metrics.recall_score(y_test, y_pred)
            p_avg += sklearn.metrics.precision_score(y_test, y_pred)
            print(this_cm) 
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

        precision.append(p_avg)
        recall.append(r_avg)
    return precision, recall

tree_counts = [10, 25, 50, 75, 100, 150, 200, 300, 500]
#tree_counts = [10, 15]

for i in range(len(tree_counts)):
    print("Running on %i trees..." % tree_counts[i])
    p, r =runForest(tree_counts[i])
    for i in range(4):
        precision[i].append(p[i])
        recall[i].append(r[i])

plt.plot(tree_counts, precision[0], label = "ecal",color='green')
plt.plot(tree_counts, precision[1], label = "ecal preshower", color='blue')
plt.plot(tree_counts, precision[2], label = "hcal", color='red')
plt.plot(tree_counts, precision[3], label = "muons", color='black')
plt.xlabel("trees")
plt.ylabel("precision")
plt.legend(loc = "upper left")
plt.title("Precision vs Number of Trees")
plt.savefig("precision.png")

#print(recall[0], recall[1], recall[2], recall[3])

plt.plot(tree_counts, recall[0], color='green', label = "ecal")
plt.plot(tree_counts, recall[1], color='blue', label = "ecal preshower")
plt.plot(tree_counts, recall[2], color='red', label = "hcal")
plt.plot(tree_counts, recall[3], color='black', label = "muons")
plt.xlabel("trees")
plt.ylabel("recall")
plt.title("Recall vs Number of Trees")
plt.legend(loc="upper left")
plt.savefig("recall.png")
