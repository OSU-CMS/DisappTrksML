#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import sklearn
import sys
import os

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

def plotCM(truth, predictions, plotDir, outputfile = 'metricPlots.root'):
    out = r.TFile(plotDir + outputfile, "update")
    labels = ["real", "fake"]
    c1 = r.TCanvas("c1", "Confusion Matrix", 800, 800)
    h_cm = r.TH2F("h_cm", "Confusion Matrix", 2, 0, 2, 2, 0, 2)
    for i in range(len(truth)):
        h_cm.Fill(truth[i], predictions[i])
    c1.cd()
    c1.SetLogz()
    h_cm.Draw("colz text")
    h_cm.GetXaxis().SetTitle("Truth")
    h_cm.GetYaxis().SetTitle("Prediction")
    h_cm.GetXaxis().SetBinLabel(1, labels[0])
    h_cm.GetXaxis().SetBinLabel(2, labels[1])
    h_cm.GetYaxis().SetBinLabel(1, labels[0])
    h_cm.GetYaxis().SetBinLabel(2, labels[1])
    r.gStyle.SetOptStat(0000)
    h_cm.SetTitle("Confusion Matrix" + " Events: " + str(len(truth)))
    c1.SaveAs(plotDir + "ConfusionMatrix.png")
    c1.Write("c_ConfusionMatrix")
    h_cm.Write("h_confusionMatrix")
    out.Close()

def getStats(truth, predictions):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(truth)):
        if(truth[i] == 0 and predictions[i] == 0): TN += 1
        if(truth[i] == 1 and predictions[i] == 1): TP += 1
        if(truth[i] == 1 and predictions[i] == 0): FN += 1
        if(truth[i] == 0 and predictions[i] == 1): FP += 1
    if(FP > 0): P = float(TP) / float((TP+FP))
    else: P = float(TP+1) / float((TP+FP+2))
    if(FN > 0): R = float(TP) / float((TP+FN))
    else: R = float(TP+1) / float((TP+FN+2))
    print("Precision (TP/(TP+FP)): " + str(P) + " Recall (TP/(TP+FN)): " + str(R))
    print("TP: " + str(TP) + ", FP: " + str(FP) + ", TN: " + str(TN) + ", FN: " + str(FN))


#import data
def loadData(dataDir):
    #layerId, charge, isPixel, pixelHitSize, pixelHitSizeX, pixelHitSizeY, stripShapeSelection, hitPosX, hitPosY
    # load the dataset
    file_count = 0
    realTracks = []
    fakeTracks = []
    for filename in os.listdir(dataDir):
        print("Loading...", dataDir + filename)
        if file_count > 20: break
        myfile = np.load(dataDir+filename)
        fakes = np.array(myfile["fake_infos"])
        reals = np.array(myfile["real_infos"])
        if(file_count == 0):
            fakeTracks = fakes
            realTracks = reals
        elif(file_count != 0 and len(fakeTracks) == 0): fakeTracks = fakes
        elif(file_count != 0 and len(realTracks) == 0): realTracks = reals
        else:
            if(len(fakes)!=0): fakeTracks = np.concatenate((fakeTracks, fakes))
            if(len(reals)!=0): realTracks = np.concatenate((realTracks, reals))
        file_count += 1


    print("Number of fake tracks:", len(fakeTracks))
    print("Number of real tracks:", len(realTracks))

    trainRealTracks, testRealTracks, trainRealTruth, testRealTruth = train_test_split(realTracks, np.zeros(len(realTracks)), test_size = 0.3)
    trainFakeTracks, testFakeTracks, trainFakeTruth, testFakeTruth = train_test_split(fakeTracks, np.ones(len(fakeTracks)), test_size = 0.3)

    #testRealTracks, valRealTracks, testRealTruth, valRealTruth = train_test_split(testRealTracks, testRealTruth, test_size = 0.5)
    #testFakeTracks, valFakeTracks, testFakeTruth, valFakeTruth = train_test_split(testFakeTracks, testFakeTruth, test_size = 0.5)

    # if undersampling
    if(undersample != -1):
        num_real = len(trainRealTracks)
        num_select = int(undersample * num_real)
        ind = np.arange(num_real)
        ind = np.random.choice(ind, num_select)
        trainRealTracks = trainRealTracks[ind]
        
    #combine all data and shuffle
    trainTracks = np.concatenate((trainFakeTracks, trainRealTracks))
    testTracks = np.concatenate((testFakeTracks, testRealTracks))
    trainTruth = np.concatenate((trainFakeTruth, trainRealTruth))
    testTruth = np.concatenate((testFakeTruth, testRealTruth))
    #valTracks = np.concatenate((valFakeTracks, valRealTracks))
    #valTruth = np.concatenate((valFakeTruth, valRealTruth))

    # Apply min max scale over all data (scale set range [-1,1])
    #scaler = MinMaxScaler(feature_range=(-1,1), copy=False)
    #scaler.partial_fit(trainTracks)
    #scaler.partial_fit(testTracks)
    #scaler.partial_fit(valTracks)
    #testTracks = scaler.transform(testTracks)
    #trainTracks = scaler.transform(trainTracks)
    #valTracks = scaler.transform(valTracks)

    test_indices = np.arange(len(testTracks))
    np.random.shuffle(test_indices)
    train_indices = np.arange(len(trainTracks))
    np.random.shuffle(train_indices)
    #val_indices = np.arange(len(valTracks))
    #np.random.shuffle(val_indices)

    trainTracks = trainTracks[train_indices]
    trainTracks = np.reshape(trainTracks, (-1,input_dim))
    if(normalize_data): trainTracks = np.tanh(trainTracks)
    trainTruth = trainTruth[train_indices]

    testTracks = testTracks[test_indices]
    testTracks = np.reshape(testTracks, (-1,input_dim))
    if(normalize_data): testTracks = np.tanh(testTracks)
    testTruth = testTruth[test_indices]

    #valTracks = valTracks[val_indices]
    #valTracks = np.reshape(valTracks, (-1,input_dim))
    #if(normalize_data): valTracks = np.tanh(valTracks)
    #valTruth = valTruth[val_indices]

    return trainTracks, testTracks, trainTruth, testTruth


def runForest(trees):
    score_avg = 0
    r_avg = 0
    p_avg = 0
    for counter in range(iterations):
        trainTracks, testTracks, trainTruth, testTruth = loadData(dataDir)  
        fake_count = 0
        for j in range(len(testTruth)):
            if(testTruth[j] == 1): fake_count += 1
        print("number of fake tracks in test group: %i" % fake_count)
        model = RandomForestClassifier(n_estimators=trees, criterion='entropy', random_state=0)
        model.fit(trainTracks, trainTruth)

        predictions = model.predict(testTracks)

        this_cm = np.array(confusion_matrix(testTruth, predictions))
        score = model.score(testTracks, testTruth)
        score_avg += score
        r_avg += sklearn.metrics.recall_score(testTruth, predictions)
        p_avg += sklearn.metrics.precision_score(testTruth, predictions)
        print(this_cm) 
        #plotCM(testTruth, predictions, '')
        if(counter == iterations-1):
            score_avg = score_avg/iterations
            r_avg = r_avg/iterations
            p_avg = p_avg/iterations
            print("average model score: " + str(score_avg) + " in " + str(iterations) + " trials \n"
                  "average precision: " + str(p_avg) + "\n"
                  "average recall: " + str(r_avg))
        del trainTracks
        del testTracks
        del trainTruth
        del testTruth
    precision = p_avg
    recall = r_avg
    return precision, recall

workDir = ''
dataDir = '/store/user/mcarrigan/fakeTracks/converted_aMC_4PlusLayer_v7p1/'
saveDir = 'saveDir/'

undersample = -1
input_dim = 163
normalize_data = False

tree_counts = [10, 25, 50, 75, 100, 150, 200, 300]
colors = ['red', 'green', 'blue', 'black', 'purple', 'pink', 'yellow', 'orange', 'cyan']
precision = []
recall = []

for i in range(len(tree_counts)):
    print("Running on %i trees..." % tree_counts[i])
    p, r =runForest(tree_counts[i])
    precision.append(p)
    recall.append(r)

plt.plot(tree_counts, precision)
plt.xlabel("trees")
plt.ylabel("precision")
plt.legend(loc = "upper left")
plt.title("Precision vs Number of Trees")
plt.savefig("precision.png")


plt.plot(tree_counts, recall)
plt.xlabel("trees")
plt.ylabel("recall")
plt.title("Recall vs Number of Trees")
plt.legend(loc="upper left")
plt.savefig("recall.png")
