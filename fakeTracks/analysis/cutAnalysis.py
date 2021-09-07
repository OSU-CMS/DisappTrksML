import matplotlib.pyplot as plt
import ROOT as r
import math
import numpy as np
from ROOT import gROOT
import os
import matplotlib

import sys

gROOT.ProcessLine('.L Infos.h++')
gROOT.SetBatch(1)

varDict = {'trackIso':0, 'eta':1, 'phi':2, 'nPV':3, 'drMinJet':4, 'ecalo':5, 'pt':6, 'd0':7, 'dz':8, 'charge':9, 'pixelHits':10, 'hits':11, 'missingOuter':12, 'dEdxPixel':13, 'dEdxStrip':14, 'pixelMeasurements':15, 'stripMeasurements':16, 'pixelSat':17, 'stripSat':18, 'sumEnergy':19, 'diffEnergy':20, 'dz1':21, 'd01':22, 'dz2':23, 'd02':24, 'dz3':25, 'd03':26, 'layer1':27, 'charge1':28, 'subDet1':29, 'pixelHitSize1':30, 'pixelHitSizeX1':31, 'pixelHitSizeY1':32, 'stripSelection1':33, 'hitPosX1':34, 'hitPosY1':35, 'layer2':36, 'charge2':37, 'subDet2':38, 'pixelHitSize2':39, 'pixelHitSizeX2':40, 'pixelHitSizeY2':41, 'stripSelection2':42, 'hitPosX2':43, 'hitPosY2':44, 'layer3':45, 'charge3':46, 'subDet3':47, 'pixelHitSize3':48, 'pixelHitSizeX3':49, 'pixelHitSizeY3':50, 'stripSelection3':51, 'hitPosX3':52, 'hitPosY3':53, 'layer4':54, 'charge4':55, 'subDet4':56, 'pixelHitSize4':57, 'pixelHitSizeX4':58, 'pixelHitSizeY4':59, 'stripSelection4':60, 'hitPosX4':61, 'hitPosY4':62, 'layer5':63, 'charge5':64, 'subDet5':65, 'pixelHitSize5':66, 'pixelHitSizeX5':67, 'pixelHitSizeY5':68, 'stripSelection5':69, 'hitPosX5':70, 'hitPosY5':71, 'layer6':72, 'charge6':73, 'subDet6':74, 'pixelHitSize6':75, 'pixelHitSizeX6':76, 'pixelHitSizeY6':77, 'stripSelection6':78, 'hitPosX6':79, 'hitPosY6':80}


def makeCuts(track, noCut):
    if(noCut!=0): 
        if(abs(track.eta) > 2.4): 
            #print("Failed eta cut")
            return False;
    if(noCut!=1): 
        if(track.pt < 55): 
            #print("Failed pt cut")
            return False;
    if(noCut!=2): 
        if(track.inGap): 
            #print("Failed inGap cut")
            return False;
    if(noCut!=3): 
        if(track.nValidPixelHits < 4): 
            #print("Failed pixel hits cut")
            return False;
    if(noCut!=4): 
        if(track.nValidHits < 4): 
            #print("Failed hits cut")
            return False;
    if(noCut!=5): 
        if(track.missingInnerHits != 0): 
            #print("Failed missing inner hits cut")
            return False;
    if(noCut!=6): 
        if(track.missingMiddleHits != 0): 
            #print("Failed missing middle hits cut")
            return False;
    #print("Found a good event", noCut) 
    return True

def nMinusPlots(dataDir, fout):

    h_eta = r.TH1F('h_eta', 'Eta of Tracks Passing N-1 Cuts', 1000, -4, 4)
    h_pt = r.TH1F('h_pt', 'Pt of Tracks Passing N-1 Cuts', 1000, 0, 1000)
    h_inGap = r.TH1F('h_inGap', 'InGap for Tracks Passing N-1 Cuts', 2, 0, 2)
    h_nValidPixelHits = r.TH1F('h_nValidPixelHits', 'Number of Valid Pixel Hits for Tracks Passing N-1 Cuts',10, 0, 10)
    h_nValidHits = r.TH1F('h_nValidHits', 'Number of Valid Hits for Tracks Passing N-1 Cuts',40, 0, 40)
    h_missingInnerHits = r.TH1F('h_missingInnerHits', 'Number of Missing Inner Hits for Tracks Passing N-1 Cuts', 16, 0, 16)
    h_missingMiddleHits = r.TH1F('h_missingMiddleHits', 'Number of Missing Middle Hits for Tracks Passing N-1 Cuts', 16, 0, 16)

    h_eta_only = r.TH1F('h_eta_only', 'Eta of Tracks Passing Eta Cut', 1000, -4, 4)
    h_pt_only = r.TH1F('h_pt_only', 'Pt of Tracks Passing Pt Cut', 1000, 0, 1000)
    h_inGap_only = r.TH1F('h_inGap_only', 'InGap of Tracks Passing inGap Cut', 2, 0, 2)
    h_nValidPixelHits_only = r.TH1F('h_nValidPixelHits_only', 'Number of Valid Pixel Hits for Tracks Passing NValidPixelHits Cut', 10, 0, 10)
    h_nValidHits_only = r.TH1F('h_nValidHits_only', 'Number of Valid Hits for Tracks Passing nValidHits Cut', 40, 0, 40)
    h_missingInnerHits_only = r.TH1F('h_missingInnerHits_only', 'Missing Inner Hits for Tracks Passing Missing Inner Hits Cut', 16, 0, 16)
    h_missingMiddleHits_only = r.TH1F('h_missingMiddleHits_only', 'Missing Middle Hits for Tracks Passing Middle Hits Cut', 16, 0, 16)

    file_count = 0

    for filename in os.listdir(dataDir):
    
        #if(file_count > 10): break
        if '.root' not in filename: continue
        file_count += 1
        fin = r.TFile(dataDir+filename, 'read')
        mytree = fin.Get('trackImageProducer/tree')
        
        print("Opening file " + dataDir + filename + ", with " + str(mytree.GetEntries()) + " events")

        for iEvent, event in enumerate(mytree):
            for iTrack, track in enumerate(event.tracks):
                #print("eta:", track.eta, "pt:", track.pt, "track.inGap", track.inGap, "pixelHits:", track.nValidPixelHits, "hits:", track.nValidHits, "missing inner hits:", track.missingInnerHits, "missing middle hits:", track.missingMiddleHits)
                if(makeCuts(track, 0)): h_eta.Fill(track.eta)
                if(makeCuts(track, 1)): h_pt.Fill(track.pt)
                if(makeCuts(track, 2)): h_inGap.Fill(track.inGap)
                if(makeCuts(track, 3)): h_nValidPixelHits.Fill(track.nValidPixelHits)
                if(makeCuts(track, 4)): h_nValidHits.Fill(track.nValidHits)
                if(makeCuts(track, 5)): h_missingInnerHits.Fill(track.missingInnerHits)
                if(makeCuts(track, 6)): h_missingMiddleHits.Fill(track.missingMiddleHits)
                if(abs(track.eta) <= 2.4): h_eta_only.Fill(track.eta)
                if(track.pt >= 55): h_pt_only.Fill(track.pt)
                if(track.inGap==False): h_inGap_only.Fill(track.inGap)
                if(track.nValidPixelHits >= 4): h_nValidPixelHits_only.Fill(track.nValidPixelHits)
                if(track.nValidHits >= 4): h_nValidHits_only.Fill(track.nValidHits)
                if(track.missingInnerHits == 0): h_missingInnerHits_only.Fill(track.missingInnerHits)
                if(track.missingMiddleHits == 0): h_missingMiddleHits_only.Fill(track.missingMiddleHits)

        

    fout.cd()
    h_eta.Write("h_eta")
    h_pt.Write("h_pt")
    h_inGap.Write("h_inGap")
    h_nValidPixelHits.Write("h_nValidPixelHits")
    h_nValidHits.Write("h_nValidHits")
    h_missingInnerHits.Write("h_missingInnerHits")
    h_missingMiddleHits.Write("h_missingMiddleHits")
    h_eta_only.Write("h_eta_only")
    h_pt_only.Write("h_pt_only")
    h_inGap_only.Write("h_inGap_only")
    h_nValidPixelHits_only.Write("h_nValidPixelHits_only")
    h_nValidHits_only.Write("h_nValidHits_only")
    h_missingInnerHits_only.Write("h_missingInnerHits_only")
    h_missingMiddleHits_only.Write("h_missingMiddleHits_only")



def nPVFakeEff(dataDir, fout):

    bins = 20

    h_nPVFake = r.TH1F("h_nPVFake", "nPV of Tracks Labelled Fake", bins, 0, 80)
    h_nPVReal = r.TH1F("h_nPVReal", "nPV of Tracks Labelled Real", bins, 0, 80)
    h_nPVPileup = r.TH1F("h_nPVPileup", "nPV of Tracks Labelled Pileup", bins, 0, 80)
    h_nPVSum = r.TH1F("h_nPVSum", "nPV of all Tracks", bins, 0, 80)

    file_count = 0

    real_count, fake_count, pileup_count = 0, 0, 0

    for filename in os.listdir(dataDir):
        
        #if(file_count > 10): break
        if '.npz' not in filename: continue
        file_count += 1

        fin = np.load(dataDir + filename)
        reals = fin['real_infos']
        fakes = fin['fake_infos']
        pileup = fin['pileup_infos']

        print(filename, "reals:", len(reals), "fakes:", len(fakes), "pileup:", len(pileup))

        for track in reals:
            h_nPVReal.Fill(track[varDict['nPV']])
            real_count += 1
        for track in fakes:
            h_nPVFake.Fill(track[varDict['nPV']])
            fake_count += 1
        for track in pileup:
            h_nPVPileup.Fill(track[varDict['nPV']])
            pileup_count += 1


    print("reals:", real_count, "fakes:", fake_count, "pileup:", pileup_count)

    h_nPVFake.Sumw2()
    h_nPVReal.Sumw2()
    h_nPVPileup.Sumw2()

    h_nPVSum.Add(h_nPVReal, h_nPVPileup)
    h_nPVSum.Add(h_nPVSum, h_nPVFake)

    h_fakeEff = r.TEfficiency(h_nPVFake, h_nPVSum)
    h_fakeEff.SetTitle("Efficiency of Predicting Fakes; nPV")
    fout.cd()
    h_fakeEff.Write("h_fakeEff")

def pileupMatching(track, event):
    min_dz = 10e6
    for vertex in event.pileupZPosition:
        dZ = abs(track.vz - vertex)
        if dZ < min_dz: min_dz = dZ
    if(min_dz == 10e6): min_dz = -1
    return min_dz

def pileupCut(dataDir, fout):

    h_min_dz = r.TH1F("h_min_dz", "Min dz Between Track and Pileup", 1000, 0, 10)
    h_dzIntegral = r.TH1F("h_dzIntegral", "Number of Events Passing dz Cut", 1000, 0, 10)
    h_dzEff = r.TH1F("h_dzEff", "Efficiency of Tracks Passing dz Cut", 1000, 0, 10)

    file_count = 0

    for filename in os.listdir(dataDir):
    
        #if(file_count > 10): break
        if '.root' not in filename: continue
        file_count += 1
        fin = r.TFile(dataDir+filename, 'read')
        mytree = fin.Get('trackImageProducer/tree')
        
        print("Opening file " + dataDir + filename + ", with " + str(mytree.GetEntries()) + " events")

        for iEvent, event in enumerate(mytree):
            for iTrack, track in enumerate(event.tracks):
                min_dz = pileupMatching(track, event)
                h_min_dz.Fill(min_dz)

    total_tracks = h_min_dz.Integral(0, 1000)

    for ibin in range(1000):
        tracks = h_min_dz.Integral(0, ibin)
        print(float(ibin)/100., tracks)
        h_dzIntegral.SetBinContent(ibin, tracks)
        h_dzEff.SetBinContent(ibin, float(tracks)/total_tracks)

    fout.cd()
    h_min_dz.Write("h_min_dz")
    h_dzIntegral.Write("h_dzIntegral")
    h_dzEff.Write("h_dzEff")


if __name__ == "__main__":

    dataDir = '/store/user/mcarrigan/Images-v8-NeutrinoGun-MC2017-ext/'
    #dataDir = '/store/user/mcarrigan/Images-v8-DYJets-MC2017_aMCNLO/'

    #procDataDir = '/store/user/mcarrigan/fakeTracks/converted_aMC_cat0p1_v2_4PlusLayer_v8p1/'
    procDataDir = '/store/user/mcarrigan/fakeTracks/converted_NeutrinoGun_cat0p1_4PlusLayer_v8p1/'

    fout = r.TFile('cutAnalysis_NG.root', 'recreate')

    nMinusPlots(dataDir, fout)

    nPVFakeEff(procDataDir, fout)

    pileupCut(dataDir, fout)



    fout.Close()








