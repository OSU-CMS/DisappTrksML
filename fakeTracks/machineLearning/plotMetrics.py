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
import utilities

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

r.gROOT.SetBatch(1)

#if len(sys.argv) > 1:
#    inputFile = sys.argv[1]
#    inputData = np.load(inputFile)
#    inputTruth = inputData["truth"]
#    inputPredictions = inputData["predictions"]
#    inputData = inputData["tracks"]

#list of Track input labels
labels = ['trackIso', 'eta', 'phi', 'nPV', 'dRMinJet', 'Ecalo', 'Pt', 'd0', 'dZ', 'Charge', 'nValidPixelHits', 'nValidHits', 'missingOuterHits', 'dEdxPixel', 'dEdxStrip', 'numMeasurementsPixel', 'numMeasurementsStrip',
          'numSatMeasurementsPixel', 'numSatMeasurementsStrip', 'sumEnergy', 'diffEnergy', 'dz1', 'd01', 'dz2', 'd02', 'dz3', 'd03',
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


def getStats(truth, predictions, plotDir, plot = False, threshold = 0.5, outputfile = 'metricPlots.root'):

    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(truth)):
        if(truth[i] == 0 and predictions[i] < threshold): TN += 1
        if(truth[i] == 1 and predictions[i] >= threshold): TP += 1
        if(truth[i] == 1 and predictions[i] < threshold): FN += 1
        if(truth[i] == 0 and predictions[i] >= threshold): FP += 1
    if(TP > 0): 
        P = float(TP) / float((TP+FP))
        R = float(TP) / float((TP+FN))
    else: 
        P = 0
        R = 0
    if(threshold == 0.5): 
        print("Precision (TP/(TP+FP)): " + str(P) + " Recall (TP/(TP+FN)): " + str(R))
        print("TP: " + str(TP) + ", FP: " + str(FP) + ", TN: " + str(TN) + ", FN: " + str(FN))       

    if(plot):
        out = r.TFile(plotDir + outputfile, "update")
        h_id = r.TH1F("h_id", "Predictions vs Truth", 4, 0, 4)
        h_score = r.TH1F("h_score", "Metric Score", 2, 0, 2)
        h_id.Fill(0, TN)
        h_id.Fill(1, TP)
        h_id.Fill(2, FN)
        h_id.Fill(3, FP)
        h_id.SetTitle("Classifications")
        h_id.GetXaxis().SetBinLabel(1, 'TN')
        h_id.GetXaxis().SetBinLabel(2, 'TP')
        h_id.GetXaxis().SetBinLabel(3, 'FN')
        h_id.GetXaxis().SetBinLabel(4, 'FP')
        h_id.GetXaxis().SetTitle('Classification')
        h_id.GetYaxis().SetTitle('Number of Tracks')
        h_id.Write("h_classification")
        h_score.Fill(0, P)
        h_score.Fill(1, R)
        h_score.SetTitle("Metric Scores")
        h_score.GetXaxis().SetBinLabel(1, 'Precision')
        h_score.GetXaxis().SetBinLabel(2, 'Recall')
        h_score.GetXaxis().SetTitle('Metric Scores')
        h_score.GetYaxis().SetTitle('Percent')
        h_score.Write('h_scoreReference')
        out.Close()

    return [TP, FP, TN, FN, P, R]

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
    h_importance = r.TH1F("h_importance", "Feature Importance", 171, 0, 171)
    h_track = r.TH1F("h_track", "Track Permutation Importance", 27, 0, 27)
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
        if i < 27:
            h_track.Fill(i, importance[i])
            h_track.GetXaxis().SetBinLabel(i+1, labels[i])
        else:
            layer = int((i-27)/9)
            index = int((i-27)%9)
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

def plotScores(predictions, truth, name, plotDir, outputfile = 'metricPlots.root'):
    out = r.TFile(plotDir + outputfile, "update")
    h_predReal = r.TH1F("h_predReal", "Prediction Scores Real" + name, 100, 0, 1)
    h_predFake = r.TH1F("h_predFake", "Prediction Scores Fake" + name, 100, 0, 1)
    for pred, truth in zip(predictions, truth):
        if truth == 0: h_predReal.Fill(pred)
        if truth == 1: h_predFake.Fill(pred)
    h_predReal.Write("Real_Scores_"+name)
    h_predFake.Write("Fake_Scores_"+name)

    c1 = r.TCanvas("c1", "c1", 800, 800)
    l1 = r.TLegend(0.7, 0.7, 0.8, 0.8)
    l1.AddEntry(h_predReal, "Real Tracks", "l")
    l1.AddEntry(h_predFake, "Fake Tracks", "l")
    c1.cd()
    c1.SetLogy()
    h_predReal.SetLineColor(2)
    h_predReal.Draw()
    h_predFake.Draw("sames")
    l1.Draw("same")
    out.cd()
    c1.Write("PredictionScores")
    out.Close()

def predictionThreshold(predictions, truth, plotDir, outputfile = 'metricPlots.root'):
    out = r.TFile(plotDir + outputfile, "update")
    h_precision = r.TH1F("h_precision", "Precision", 20, 0, 1)
    h_recall = r.TH1F("h_recall", "Recall", 20, 0, 1)
    c1 = r.TCanvas("c1", "c1", 800, 800)

    bins = [x*0.05 for x in range(20)]

    for iBin, Bin in enumerate(bins):
        #[TP, FP, TN, FN]
        metrics = getStats(truth, predictions, plotDir, threshold = Bin)
        print(Bin, metrics)
        h_precision.SetBinContent(iBin+1, metrics[4])
        h_recall.SetBinContent(iBin+1, metrics[5])
    
    c1.cd()
    l1 = r.TLegend(0.5, 0.7, 0.6, 0.8)
    h_precision.GetYaxis().SetRangeUser(0,1)
    h_precision.SetTitle("Prediction Threshold vs Metrics")
    h_precision.GetYaxis().SetTitle("Metric Score")
    h_precision.GetXaxis().SetTitle("Prediction Threshold")
    h_precision.Draw()
    h_recall.SetLineColor(2)
    h_recall.Draw("same")
    l1.AddEntry(h_precision, "Precision", "l")
    l1.AddEntry(h_recall, "Recall", "l")
    l1.Draw("same")

    out.cd()
    h_precision.Write("PrecisionVsThreshold")
    h_recall.Write("RecallVsThreshold")
    c1.Write("PredictionThresholdScores")
    out.Close()

def backgroundEstimation(tracks, predictions, d0Index, plotDir, outputfile = 'metricPlots.root'):
    
    out = r.TFile(plotDir + outputfile, "update")
    h_d0Bkg = r.TH1F("h_d0Bkg", "d0 of Tracks Labeled Real", 100, -0.1, 0.1)
    h_d0Fake = r.TH1F("h_d0Fake" ,"d0 of Tracks Labeled Fake", 100, -0.1, 0.1)
    c1 = r.TCanvas("c1", "c1", 600, 800)

    fake, background = 0, 0

    for itrack, track in enumerate(tracks):
        if abs(tracks[itrack, d0Index]) < 0.0002:
            if predictions[itrack] > 0.5:
                fake += 1
                h_d0Fake.Fill(tracks[itrack, d0Index])
            else:
                background += 1
                h_d0Bkg.Fill(tracks[itrack, d0Index])

    
    c1.cd()
    h_d0Bkg.Draw()
    h_d0Fake.SetLineColor(2)
    h_d0Fake.Draw("same")
    l1 = r.TLegend(0.2, 0.7, 0.3, 0.9)
    l1.AddEntry(h_d0Bkg, "Labeled Real", "l")
    l1.AddEntry(h_d0Fake, "Labeled Fake", "l")
    l1.Draw("same")
    out.cd()
    h_d0Bkg.Write()
    h_d0Fake.Write()
    c1.Draw("PredictedD0")
    out.Close()
    print("Predicted background is %d out of %d events" % (fake, background+fake))

if __name__ == "__main__":

    #plotCM(inputTruth, inputPredictions)
    #getStats(inputTruth, inputPredictions)
    #plotHistory(history, ['loss','auc'])


    pred = np.load('/data/users/mcarrigan/fakeTracks_4PlusLayer_aMCv9p1_9_1_NGBoost_PUReal/fakeTracks_4PlusLayer_aMCv9p1_9_1_NGBoost_PUReal_p1/outputFiles/predictions.npz')['predictions']
    plotScores(pred, "test", '/data/users/mcarrigan/fakeTracks_4PlusLayer_aMCv9p1_9_1_NGBoost_PUReal/fakeTracks_4PlusLayer_aMCv9p1_9_1_NGBoost_PUReal_p1/plots/')


