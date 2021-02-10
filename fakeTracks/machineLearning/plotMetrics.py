import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial
from itertools import repeat
import numpy as np
import os
from collections import Counter
from sklearn.metrics import roc_auc_score
import sys
import ROOT as r


class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

saveDir = "images/"

#if len(sys.argv) > 1:
#    inputFile = sys.argv[1]
#    inputData = np.load(inputFile)
#    inputTruth = inputData["truth"]
#    inputPredictions = inputData["predictions"]
#    inputData = inputData["tracks"]


def plotCM(truth, predictions, plotDir):
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

def plotHistory(history, variables, plotDir, outputfile = 'metricPlots.root'):
    out = r.TFile(plotDir + outputfile, "recreate")
    for var in variables:
        pltName = "c_" + str(var)
        c1 = r.TCanvas(pltName, pltName, 800, 800)
        metric = history.history[var]
        g1 = r.TGraph(len(metric), np.arange(len(metric)).astype('float64'), np.array(metric).astype('float64'))
        g1.Draw()
        g1.SetTitle(var)
        g1.GetXaxis().SetTitle("Epochs")
        g1.GetYaxis().SetTitle(var)
        metricVal = history.history["val_" + var]
        g2 = r.TGraph(len(metric), np.arange(len(metricVal)).astype('float64'), np.array(metricVal).astype('float64'))
        g2.Draw("SAME")
        g2.SetLineColor(2)
        l1 = r.TLegend(0.7, 0.7, 0.8, 0.8)
        l1.AddEntry(g1, var, "l")
        l1.AddEntry(g2, "val_" + str(var), "l")
        l1.Draw()
        c1.Write(pltName)
        g1.Write(var)
        g2.Write("val_"+str(var))
    out.Close()


if __name__ == "__main__":

    plotCM(inputTruth, inputPredictions)
    getStats(inputTruth, inputPredictions)
    plotHistory(history, ['loss','auc'])


