import ROOT as r
import os
import sys
import numpy as np


def loadFromLog(logNum):
    logIn = open(directory + 'out_' + str(logNum) + '.txt')
    contents = logIn.readlines()
    contents = (contents[9].split()[3]).split(']')[0]
    return contents

#directory = '/data/users/mcarrigan/fakeTracks/networks/filterSearch/fakeTracks_4PlusLayer_aMCv9p3_11_4_NGBoost_filterSearch/'
directory = '/data/users/mcarrigan/fakeTracks/networks/dropoutSearch/fakeTracks_4PlusLayer_aMCv9p3_11_4_NGBoost_dropoutSearch_11_22/'

fout = r.TFile("outputPlots_dropoutSearch.root", "recreate")

numSearches = 0 #number of different variables being searched
repeats = 1 #number of times each variable is tried (average over mutiple trials)

if os.path.isfile(directory + 'params.npy'):
    if os.path.isfile(directory + 'jobInfo.npy'):
        params = np.load(directory+'params.npy', allow_pickle=True)
        jobInfo = np.load(directory+'jobInfo.npy', allow_pickle=True)
        numSearches = int(len(params) / jobInfo[0])
        repeats = jobInfo[0]
    else:
        numSearches = len(params)
else:
    for subdir in os.listdir(directory):
        if directory.split('/')[-2] in subdir: 
            numSearches += 1

print(numSearches, repeats)

searchParam = 'Dropout'

h_precision = r.TH1F("h_precision", "Precision", numSearches, 0, numSearches)
h_recall = r.TH1F("h_recall", "Recall", numSearches, 0, numSearches)

c1 = r.TCanvas("c1", "c1", 600, 600)

bins = []
precision = {}
recall = {}

#variable to make true if networkInfo does not have the search value (it should still be in logs and can use loadFromLog to find it
noBinNames = True

for subdir in os.listdir(directory):
    if directory.split('/')[-2] not in subdir: continue
    #print(directory.split('/')[-2], subdir)
    
    fin = open(directory + subdir + '/outputFiles/networkInfo.txt', 'r')
    bin_name = ''
    for line in fin:
        if(noBinNames and bin_name == ''):
            logNum = int(subdir.split('_p')[-1])
            bin_name = loadFromLog(logNum)
            if bin_name not in bins: bins.append(bin_name)
            print('Log Number:', logNum, 'Bin Name: ', bin_name)
        if searchParam + ':' in line:
            bin_name = line.split(':')[-1]
            if bin_name not in bins: bins.append(bin_name)
        elif 'Precision:' in line:
            if bin_name in precision.keys():
                precision[bin_name] += float(line.split(' ')[-1])
            else:
                precision[bin_name] = float(line.split(' ')[-1])
            if float(line.split(' ')[-1]) < 0.8: print("Precision:", bin_name, float(line.split(' ')[-1]))
        elif 'Recall:' in line:
            if bin_name in recall.keys():
                recall[bin_name] += float(line.split(' ')[-1])
            else:
                recall[bin_name] = float(line.split(' ')[-1])
            if float(line.split(' ')[-1]) < 0.8: print("Recall:", bin_name, float(line.split(' ')[-1]))
	
for ibin, Bin in enumerate(bins):
    h_precision.GetXaxis().SetBinLabel(ibin+1, Bin)
    h_recall.GetXaxis().SetBinLabel(ibin+1, Bin)
    h_precision.Fill(ibin, precision[Bin])
    h_recall.Fill(ibin, recall[Bin])

h_precision.Scale(1./repeats)
h_recall.Scale(1./repeats)

fout.cd()
h_precision.Write()
h_recall.Write()

c1.cd()
h_precision.SetTitle("Precision and Recall")
h_precision.GetXaxis().SetTitle(searchParam)
h_precision.GetYaxis().SetRangeUser(0,1)
h_precision.SetMarkerStyle(22)
h_precision.SetMarkerColor(4)
h_precision.Draw("hist TEXT0 p")
h_recall.SetMarkerStyle(23)
h_recall.SetMarkerColor(2)
h_recall.Draw("same hist TEXT0 p")
r.gStyle.SetOptStat(0000)
r.gStyle.SetPaintTextFormat("2.3f")
l1 = r.TLegend(0.2, 0.2, 0.4, 0.4)
l1.AddEntry(h_precision, "Precision", "p")
l1.AddEntry(h_recall, "Recall", "p")
l1.Draw()
fout.cd()
c1.Write("Metrics")
