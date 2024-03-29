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
from sklearn.inspection import permutation_importance
import sys
import ROOT as r
from ROOT.TMath import Gaus

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

#list of Track input labels
labels = ['trackIso', 'eta', 'phi', 'nPV', 'dRMinJet', 'Ecalo', 'Pt', 'd0', 'dZ', 'Charge', 'nValidPixelHits', 'nValidHits', 'missingOuterHits', 'dEdxPixel', 'dEdxStrip', 'numMeasurementsPixel', 'numMeasurementsStrip',
          'numSatMeasurementsPixel', 'numSatMeasurementsStrip', 
          'Layer1', 'Charge1', 'isPixel1', 'pixelHitSize1', 'pixelHitSizeX1', 'pixelHitSizeY1', 'stripShapeSelection1', 'hitPosX1', 'hitPoxY1',
          'Layer2', 'Charge2', 'isPixel2', 'pixelHitSize2', 'pixelHitSizeX2', 'pixelHitSizeY2', 'stripShapeSelection2', 'hitPosX2', 'hitPoxY2',
          'Layer3', 'Charge3', 'isPixel3', 'pixelHitSize3', 'pixelHitSizeX3', 'pixelHitSizeY3', 'stripShapeSelection3', 'hitPosX3', 'hitPoxY3',
          'Layer4', 'Charge4', 'isPixel4', 'pixelHitSize4', 'pixelHitSizeX4', 'pixelHitSizeY4', 'stripShapeSelection4', 'hitPosX4', 'hitPoxY4',
          'Layer5', 'Charge5', 'isPixel5', 'pixelHitSize5', 'pixelHitSizeX5', 'pixelHitSizeY5', 'stripShapeSelection5', 'hitPosX5', 'hitPoxY5',
          'Layer6', 'Charge6', 'isPixel6', 'pixelHitSize6', 'pixelHitSizeX6', 'pixelHitSizeY6', 'stripShapeSelection6', 'hitPosX6', 'hitPoxY6',
          'Layer7', 'Charge7', 'isPixel7', 'pixelHitSize7', 'pixelHitSizeX7', 'pixelHitSizeY7', 'stripShapeSelection7', 'hitPosX7', 'hitPoxY7',
          'Layer8', 'Charge8', 'isPixel8', 'pixelHitSize8', 'pixelHitSizeX8', 'pixelHitSizeY8', 'stripShapeSelection8', 'hitPosX8', 'hitPoxY8',
          'Layer9', 'Charge9', 'isPixel9', 'pixelHitSize9', 'pixelHitSizeX9', 'pixelHitSizeY9', 'stripShapeSelection9', 'hitPosX9', 'hitPoxY9',
          'Layer10', 'Charge10', 'isPixel10', 'pixelHitSize10', 'pixelHitSizeX10', 'pixelHitSizeY10', 'stripShapeSelection10', 'hitPosX10', 'hitPoxY10',
          'Layer11', 'Charge11', 'isPixel11', 'pixelHitSize11', 'pixelHitSizeX11', 'pixelHitSizeY11', 'stripShapeSelection11', 'hitPosX11', 'hitPoxY11',
          'Layer12', 'Charge12', 'isPixel12', 'pixelHitSize12', 'pixelHitSizeX12', 'pixelHitSizeY12', 'stripShapeSelection12', 'hitPosX12', 'hitPoxY12',
          'Layer13', 'Charge13', 'isPixel13', 'pixelHitSize13', 'pixelHitSizeX13', 'pixelHitSizeY13', 'stripShapeSelection13', 'hitPosX13', 'hitPoxY13',
          'Layer14', 'Charge14', 'isPixel14', 'pixelHitSize14', 'pixelHitSizeX14', 'pixelHitSizeY14', 'stripShapeSelection14', 'hitPosX14', 'hitPoxY14',
          'Layer15', 'Charge15', 'isPixel15', 'pixelHitSize15', 'pixelHitSizeX15', 'pixelHitSizeY15', 'stripShapeSelection15', 'hitPosX15', 'hitPoxY15',
          'Layer16', 'Charge16', 'isPixel16', 'pixelHitSize16', 'pixelHitSizeX16', 'pixelHitSizeY16', 'stripShapeSelection16', 'hitPosX16', 'hitPoxY16']



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

def plotCM3(predictions, plotDir, outputfile = 'metricPlots.root'):
    out = r.TFile(plotDir + outputfile, "update")
    labels = ["real", "pileup", "fake"]
    c1 = r.TCanvas("c1", "Confusion Matrix", 800, 800)
    h_cm = r.TH2F("h_cm", "Confusion Matrix", 3, 0, 3, 3, 0, 3)
    for i in range(3):
        for j in range(3):
            h_cm.Fill(j, i, predictions[j][i])
    c1.cd()
    c1.SetLogz()
    h_cm.Draw("colz text")
    h_cm.GetXaxis().SetTitle("Truth")
    h_cm.GetYaxis().SetTitle("Prediction")
    h_cm.GetXaxis().SetBinLabel(1, labels[0])
    h_cm.GetXaxis().SetBinLabel(2, labels[1])
    h_cm.GetXaxis().SetBinLabel(3, labels[2])
    h_cm.GetYaxis().SetBinLabel(1, labels[0])
    h_cm.GetYaxis().SetBinLabel(2, labels[1])
    h_cm.GetYaxis().SetBinLabel(3, labels[2])
    r.gStyle.SetOptStat(0000)
    h_cm.SetTitle("Confusion Matrix" + " Events: " + str(np.sum(predictions)))
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

def plotHistory(history, variables, plotDir, outputfile = 'metricPlots.root'):
    out = r.TFile(plotDir + outputfile, "update")
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

def permutationImportance(model, tracks, truth, plotDir, outputfile = 'metricPlots.root'):
    out = r.TFile(plotDir + outputfile, "update")
    h_importance = r.TH1F("h_importance", "Feature Importance", 163, 0, 163)
    h_track = r.TH1F("h_track", "Track Permutation Importance", 19, 0, 19)
    h_layer1 = r.TH1F("h_layer1", "Layer 1 Permutation Importance", 9, 0, 9)
    h_layer2 = r.TH1F("h_layer2", "Layer 2 Permutation Importance", 9, 0, 9)
    h_layer3 = r.TH1F("h_layer3", "Layer 3 Permutation Importance", 9, 0, 9)
    h_layer4 = r.TH1F("h_layer4", "Layer 4 Permutation Importance", 9, 0, 9)
    h_layer5 = r.TH1F("h_layer5", "Layer 5 Permutation Importance", 9, 0, 9)
    h_layer6 = r.TH1F("h_layer6", "Layer 6 Permutation Importance", 9, 0, 9)
    h_layer7 = r.TH1F("h_layer7", "Layer 7 Permutation Importance", 9, 0, 9)
    h_layer8 = r.TH1F("h_layer8", "Layer 8 Permutation Importance", 9, 0, 9)
    h_layer9 = r.TH1F("h_layer9", "Layer 9 Permutation Importance", 9, 0, 9)
    h_layer10 = r.TH1F("h_layer10", "Layer 10 Permutation Importance", 9, 0, 9)
    h_layer11 = r.TH1F("h_layer11", "Layer 11 Permutation Importance", 9, 0, 9)
    h_layer12 = r.TH1F("h_layer12", "Layer 12 Permutation Importance", 9, 0, 9)
    h_layer13 = r.TH1F("h_layer13", "Layer 13 Permutation Importance", 9, 0, 9)
    h_layer14 = r.TH1F("h_layer14", "Layer 14 Permutation Importance", 9, 0, 9)
    h_layer15 = r.TH1F("h_layer15", "Layer 15 Permutation Importance", 9, 0, 9)
    h_layer16 = r.TH1F("h_layer16", "Layer 16 Permutation Importance", 9, 0, 9)
    h_layers = [h_layer1, h_layer2, h_layer3, h_layer4, h_layer5, h_layer6, h_layer7, h_layer8, 
                h_layer9, h_layer10, h_layer11, h_layer12, h_layer13, h_layer14, h_layer15, h_layer16]
   
    results = permutation_importance(model, tracks, truth, scoring='accuracy')
    importance = results.importances_mean
    for i in range(len(importance)):
        print(i, labels[i], importance[i])
        h_importance.Fill(i, importance[i])
        h_importance.GetXaxis().SetBinLabel(i+1, labels[i])
        if i < 19:
            h_track.Fill(i, importance[i])
            h_track.GetXaxis().SetBinLabel(i+1, labels[i])
        else:
            layer = int((i-19)/9)
            index = int((i-19)%9)
            print(i, layer, index)
            h_layers[layer].Fill(index,importance[i])
            h_layers[layer].GetXaxis().SetBinLabel(index+1,labels[i])  
    h_importance.SetTitle("Permutation Feature Importance")
    h_importance.GetYaxis().SetTitle("Permutation Importance Score")
    r.gStyle.SetOptStat(0000)
    h_importance.Write("h_Importance")
    h_track.Write("h_track_importance")
    for i in range(16):
        h_layers[i].Write()
    out.Close()

def predictionCorrelation(predictions, variable, bins, min_bin, max_bin, name, plotDir, outputfile = 'metricPlots.root'):
    out = r.TFile(plotDir + outputfile, "update")
    c1 = r.TCanvas("c1", name, 800, 800)
    fake_indices = np.argwhere(predictions >= 0.5)
    fakes = variable[fake_indices[:,0]]
    print(fakes[:10])
    h1 = r.TH1F("h1", "h1", bins, min_bin, max_bin)
    for fake in fakes:
        h1.Fill(fake)
    h1.Write(name)
    c1.cd()
    h1.SetMarkerStyle(8)
    h1.Draw("P")
    f1 = r.TF1("f1", "[0]*TMath::Gaus(x, 0, [1])+[2]", -1, 1)
    f1.SetParLimits(0, 10e-3, 10e3)
    h1.Fit(f1, "LQMR", "", 0.1, 1)
    f1.Draw()
    h1.Draw("same p")
    c1.Write('c_' + name)
    out.Close()

def comparePredictions(predictions, variable, bins, min_bin, max_bin, name, plotDir, outputfile = 'metricPlots.root'):
    out = r.TFile(plotDir + outputfile, "update")
    c1 = r.TCanvas("c1", name, 800, 800)
    fake_indices = np.argwhere(predictions >= 0.5)
    fakes = variable[fake_indices[:,0]]
    real_indices = np.argwhere(predictions < 0.5)
    reals = variable[real_indices[:,0]]
    h_fake = r.TH1F("h_fake", name+"Predicted Fakes", bins, min_bin, max_bin)
    h_real = r.TH1F("h_real", name+"Predicted Real", bins, min_bin, max_bin) 
    h_total = r.TH1F("h_total", name, bins, min_bin, max_bin)
    for fake in fakes:
        h_fake.Fill(fake)
    for real in reals:
        h_real.Fill(real)
    for i in variable:
        h_total.Fill(i)

    h_eff = r.TEfficiency(h_real, h_total)
    c1.cd()
    h_fake.Draw()
    h_real.SetLineColor(2)
    h_real.Draw("same")
    l1 = r.TLegend(0.7, 0.8, 0.9, 0.9)
    l1.AddEntry(h_real, "Real: " + str(len(reals)), "l")
    l1.AddEntry(h_fake, "Fake: " + str(len(fakes)), "l")
    l1.Draw()
    r.gStyle.SetOptStat(0000)
    c1.Write("c_" + name)
    h_fake.Write("fake_"+name)
    h_real.Write("real_"+name)
    h_eff.Write("efficiency_" + name)
    out.Close()

def plotScores(predictions, name, plotDir, outputfile = 'metricPlots.root'):
    out = r.TFile(plotDir + outputfile, "update")
    h_pred = r.TH1F("h_pred", "Prediction Scores" + name, 100, 0, 1)
    for pred in predictions:
        h_pred.Fill(pred)
    h_pred.Write("Scores_"+name)
    out.Close()

if __name__ == "__main__":

    plotCM(inputTruth, inputPredictions)
    getStats(inputTruth, inputPredictions)
    plotHistory(history, ['loss','auc'])


