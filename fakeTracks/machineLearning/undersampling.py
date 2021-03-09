import numpy as np
import ROOT as r
import os
import random
import sys
import pandas as pd

dataDir = '/store/user/mcarrigan/fakeTracks/'

dataSets = ['converted_madgraph_6layer_v7p1', 'converted_aMC_6layer_v7p1']

out = r.TFile("Correlations_6Layer.root", "recreate")

input_dim = 163

undersamples = [0.9, 0.8, 0.6, 0.4, 0.2]
undersample_title = ['0p9', '0p8', '0p6', '0p4', '0p2']

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

for i, thisDataset in enumerate(dataSets):
    dataLoc = dataDir + thisDataset + '/'
    file_count = 0
    for filename in os.listdir(dataLoc):
        #if(file_count > 20): continue
        print("Loading... " + dataLoc + filename)
        if '.npz' and 'events' in filename:
            this_data = np.load(dataLoc + filename)
            if(i == 0 and file_count == 0): 
                real_data = this_data['real_infos']
                fake_data = this_data['fake_infos']
            else:
                if len(this_data['real_infos']) != 0: real_data = np.vstack((real_data, this_data['real_infos']))
                if len(this_data['fake_infos']) != 0: fake_data = np.vstack((fake_data, this_data['fake_infos']))
        file_count += 1

print("Number of real tracks: " + str(real_data.shape[0]))
print("Number of fake tracks: " + str(fake_data.shape[0]))

h_corr = r.TH2F("h_corr", "Correlation of Data", input_dim, 0, input_dim, input_dim, 0, input_dim)
h_RealCorr = r.TH2F("h_RealCorr", "Correlation of Real Data", input_dim, 0, input_dim, input_dim, 0, input_dim)
h_FakeCorr = r.TH2F("h_FakeCorr", "Correlation of Fake Data", input_dim, 0, input_dim, input_dim, 0, input_dim)

c1 = r.TCanvas("c1", "Correlation of Data", 800, 800)

all_data = np.vstack((real_data, fake_data))
np.random.shuffle(all_data)
corr_matrix = np.corrcoef(all_data, rowvar=False)

np.random.shuffle(real_data)
m_realCorr = np.corrcoef(real_data, rowvar=False)

np.random.shuffle(fake_data)
m_fakeCorr = np.corrcoef(fake_data, rowvar=False)

print(corr_matrix.shape)
#print(corr_matrix)

for i in range(input_dim):
    h_corr.GetXaxis().SetBinLabel(i+1, labels[i])
    h_RealCorr.GetXaxis().SetBinLabel(i+1, labels[i])
    h_FakeCorr.GetXaxis().SetBinLabel(i+1, labels[i])
    for j in range(input_dim):
        h_corr.Fill(i, j, corr_matrix[i, j])
        h_RealCorr.Fill(i, j, m_realCorr[i, j])
        h_FakeCorr.Fill(i, j, m_fakeCorr[i, j])
    if(i==0):
        h_corr.GetYaxis().SetBinLabel(j+1, labels[j])
        h_RealCorr.GetYaxis().SetBinLabel(j+1, labels[j])
        h_FakeCorr.GetYaxis().SetBinLabel(j+1, labels[j])

h_corr.Write()
h_RealCorr.Write()
h_FakeCorr.Write()

for under in range(len(undersamples)):
    h_corr.Reset()
    num_reals = len(real_data)
    ind = np.arange(num_reals)
    num_under = int(undersamples[under] * num_reals)
    ind = np.random.choice(ind, num_under)
    under_data = real_data[ind]
    all_data = np.vstack((under_data, fake_data))
    np.random.shuffle(all_data)
    corr_matrix = np.corrcoef(all_data, rowvar=False)
    for i in range(input_dim):
        h_corr.GetXaxis().SetBinLabel(i+1, labels[i])
        for j in range(input_dim):
            h_corr.GetYaxis().SetBinLabel(j+1, labels[j])
            h_corr.Fill(i, j, corr_matrix[i, j])
    h_corr.Write("h_corr_" + undersample_title[under])

out.Close()

