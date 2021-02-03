import matplotlib.pyplot as plt
import ROOT as r
import math
import numpy as np
from ROOT import gROOT
from ROOT.Math import XYZVector
import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import sys

gROOT.ProcessLine('.L /share/scratch0/mcarrigan/disTracksML/CMSSW_9_4_9/src/DisappTrksML/TreeMaker/interface/Infos.h++')
gROOT.SetBatch()

imageDir = '/store/user/mcarrigan/Images-v6-DYJets-MC2017/'
imgOutDir = 'images/'

#file for recording fake tracks fo fireworks displays
event_list = open("event_list.txt", "a")

#fin = r.TFile(imageDir + 'images_1.root')
fout = r.TFile('analysisPlots.root', 'RECREATE')
#tree = fin.Get('trackImageProducer/tree')
#print("Opened file",fname)
#nEvents = tree.GetEntries()
#if(nEvents == 0):
#    sys.exit("0 events found in file")
#print("Added",nEvents)

res_eta = 50
res_phi = 50
eta_ub,eta_lb = 0.25,-0.25
phi_ub,phi_lb = 0.25,-0.25

trkRes_Phi = 80
trkRes_Eta = 80
trkPhi_ub, trkPhi_lb = 4, -4
trkEta_ub, trkEta_lb = 4, -4 

trkRes_x = 100
trkRes_y = 100
trkX_ub, trkX_lb = 50, -50
trkY_ub, trkY_lb = 50, -50

dR_cut = 0.15

OverUnderFlow = False

def convert_eta(eta):
        return int(round(((res_eta-1)*1.0/(eta_ub-eta_lb))*(eta-eta_lb)))

def convert_phi(phi):
        return int(round(((res_phi-1)*1.0/(phi_ub-phi_lb))*(phi-phi_lb)))

def convert_coord(x):
        return int(round(0.5*x+25))

# combine EB+EE and muon detectors into ECAL/HCAL/MUO indices
def type_to_channel(hittype):
    #none
    if(hittype == 0): return -1
    #ECAL (EE,EB)
    if(hittype == 1 or hittype == 2): return 0
    #ES
    if(hittype == 3): return 1
    #HCAL
    if(hittype == 4): return 2
    #Muon (CSC,DT,RPC)
    if(hittype == 5 or hittype ==6 or hittype == 7): return 3

# return (dEta, dPhi) between track and hit
def imageCoordinates(track, hit):
        momentum = XYZVector(track.px, track.py, track.pz)
        track_eta = momentum.Eta()
        track_phi = momentum.Phi()
        dEta = track_eta - hit.eta
        dPhi = track_phi - hit.phi
        # branch cut [-pi, pi)
        if abs(dPhi) > math.pi:
                dPhi -= round(dPhi / (2. * math.pi)) * 2. * math.pi
        return (dEta, dPhi)


def getMaxLayerHits(layers, charges):
    max_layers, max_charges = [], []
    charges = np.stack((layers, charges), axis=-1)
    indices = np.lexsort((charges[:,1], charges[:,0]))
    charges = charges[indices]
    for i in range(len(charges)):
        if charges[i, 0] not in max_layers: 
            max_layers.append(charges[i,0])
            max_charges.append(charges[i,1])
        else:
            idx = max_layers.index(charges[i,0])
            if charges[i,1] > max_charges[idx]: max_charges[idx] = charges[i,1]
    return max_layers, max_charges

fakeCount = 0
muonCount = 0

hist_dR = r.TH1F("histdR", "histdR", 700, 0, 7)
h_d0Fake = r.TH1F("d0Fake", "d0 for dR > " + str(dR_cut), 80, -40, 40)
h_d0Other = r.TH1F("d0Other", "d0 for dR < " + str(dR_cut), 80, -40, 40)
h_dZFake = r.TH1F("dZFake", "dZ for dR > " + str(dR_cut), 200, -100, 100)
h_dZOther = r.TH1F("dZOther", "dZ for dR < " + str(dR_cut), 200, -100, 100)

h_ECAL = r.TH2F("hECAL", "ECAL", res_eta, eta_lb, eta_ub, res_phi, phi_lb, phi_ub)
h_preS = r.TH2F("hPreS", "Pre Shower", res_eta, eta_lb, eta_ub, res_phi, phi_lb, phi_ub)
h_HCAL = r.TH2F("hHCAL", "HCAL", res_eta, eta_lb, eta_ub, res_phi, phi_lb, phi_ub)
h_MS = r.TH2F("hMS", "Muon System", res_eta, eta_lb, eta_ub, res_phi, phi_lb, phi_ub)

h_Calo = [h_ECAL, h_preS, h_HCAL, h_MS]
caloTitles = ["ECAL", "Pre Shower", "HCAL", "MS"]

h_Trk1 = r.TH2F("hTrk1", "Fake Layer 1", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_Trk2 = r.TH2F("hTrk2", "Fake Layer 2", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_Trk3 = r.TH2F("hTrk3", "Fake Layer 3", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_Trk4 = r.TH2F("hTrk4", "Fake Layer 4", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_Trk5 = r.TH2F("hTrk5", "Fake Layer 5", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_Trk6 = r.TH2F("hTrk6", "Fake Layer 6", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_Trk7 = r.TH2F("hTrk7", "Fake Layer 7", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_Trk8 = r.TH2F("hTrk8", "Fake Layer 8", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_Trk9 = r.TH2F("hTrk9", "Fake Layer 9", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_Trk10 = r.TH2F("hTrk10", "Fake Layer 10", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_Trk11 = r.TH2F("hTrk11", "Fake Layer 11", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_Trk12 = r.TH2F("hTrk12", "Fake Layer 12", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_Trk13 = r.TH2F("hTrk13", "Fake Layer 13", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_Trk14 = r.TH2F("hTrk14", "Fake Layer 14", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_Trk15 = r.TH2F("hTrk15", "Fake Layer 15", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_Trk16 = r.TH2F("hTrk16", "Fake Layer 16", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)

h_Tracker = [h_Trk1, h_Trk2, h_Trk3, h_Trk4, h_Trk5, h_Trk6, h_Trk7, h_Trk8, h_Trk9, h_Trk10, h_Trk11, h_Trk12, h_Trk13, h_Trk14, h_Trk15, h_Trk16]

h_muTrk1 = r.TH2F("muTrk1", "Muon Layer 1", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_muTrk2 = r.TH2F("muTrk2", "Muon Layer 2", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_muTrk3 = r.TH2F("muTrk3", "Muon Layer 3", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_muTrk4 = r.TH2F("muTrk4", "Muon Layer 4", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_muTrk5 = r.TH2F("muTrk5", "Muon Layer 5", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_muTrk6 = r.TH2F("muTrk6", "Muon Layer 6", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_muTrk7 = r.TH2F("muTrk7", "Muon Layer 7", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_muTrk8 = r.TH2F("muTrk8", "Muon Layer 8", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_muTrk9 = r.TH2F("muTrk9", "Muon Layer 9", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_muTrk10 = r.TH2F("muTrk10", "Muon Layer 10", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_muTrk11 = r.TH2F("muTrk11", "Muon Layer 11", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_muTrk12 = r.TH2F("muTrk12", "Muon Layer 12", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_muTrk13 = r.TH2F("muTrk13", "Muon Layer 13", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_muTrk14 = r.TH2F("muTrk14", "Muon Layer 14", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_muTrk15 = r.TH2F("muTrk15", "Muon Layer 15", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)
h_muTrk16 = r.TH2F("muTrk16", "Muon Layer 16", trkRes_x, trkX_lb, trkX_ub, trkRes_y, trkY_lb, trkY_ub)

h_MuonTrk = [h_muTrk1, h_muTrk2, h_muTrk3, h_muTrk4, h_muTrk5, h_muTrk6, h_muTrk7, h_muTrk8, h_muTrk9, h_muTrk10, h_muTrk11, h_muTrk12, h_muTrk13, h_muTrk14, h_muTrk15, h_muTrk16]

h_fake_dedx1 = r.TH1F("hdedx1", "Fake DeDx Layer 1", 120, 0, 60)
h_fake_dedx2 = r.TH1F("hdedx2", "Fake DeDx Layer 2", 120, 0, 60)
h_fake_dedx3 = r.TH1F("hdedx3", "Fake DeDx Layer 3", 120, 0, 60)
h_fake_dedx4 = r.TH1F("hdedx4", "Fake DeDx Layer 4", 120, 0, 60)
h_fake_dedx5 = r.TH1F("hdedx5", "Fake DeDx Layer 5", 120, 0, 60)
h_fake_dedx6 = r.TH1F("hdedx6", "Fake DeDx Layer 6", 120, 0, 60)
h_fake_dedx7 = r.TH1F("hdedx7", "Fake DeDx Layer 7", 120, 0, 60)
h_fake_dedx8 = r.TH1F("hdedx8", "Fake DeDx Layer 8", 120, 0, 60)
h_fake_dedx9 = r.TH1F("hdedx9", "Fake DeDx Layer 9", 120, 0, 60)
h_fake_dedx10 = r.TH1F("hdedx10", "Fake DeDx Layer 10", 120, 0, 60)
h_fake_dedx11 = r.TH1F("hdedx11", "Fake DeDx Layer 11", 120, 0, 60)
h_fake_dedx12 = r.TH1F("hdedx12", "Fake DeDx Layer 12", 120, 0, 60)
h_fake_dedx13 = r.TH1F("hdedx13", "Fake DeDx Layer 13", 120, 0, 60)
h_fake_dedx14 = r.TH1F("hdedx14", "Fake DeDx Layer 14", 120, 0, 60)
h_fake_dedx15 = r.TH1F("hdedx15", "Fake DeDx Layer 15", 120, 0, 60)
h_fake_dedx16 = r.TH1F("hdedx16", "Fake DeDx Layer 16", 120, 0, 60)

h_fake_dedx = [h_fake_dedx1, h_fake_dedx2, h_fake_dedx3, h_fake_dedx4, h_fake_dedx5, h_fake_dedx6, h_fake_dedx7, h_fake_dedx8, h_fake_dedx9, h_fake_dedx10, h_fake_dedx11, h_fake_dedx12, h_fake_dedx13, h_fake_dedx14, h_fake_dedx15, h_fake_dedx16] 

h_muon_dedx1 = r.TH1F("mdedx1", "Muon DeDx Layer 1", 120, 0, 60)
h_muon_dedx2 = r.TH1F("mdedx2", "Muon DeDx Layer 2", 120, 0, 60)
h_muon_dedx3 = r.TH1F("mdedx3", "Muon DeDx Layer 3", 120, 0, 60)
h_muon_dedx4 = r.TH1F("mdedx4", "Muon DeDx Layer 4", 120, 0, 60)
h_muon_dedx5 = r.TH1F("mdedx5", "Muon DeDx Layer 5", 120, 0, 60)
h_muon_dedx6 = r.TH1F("mdedx6", "Muon DeDx Layer 6", 120, 0, 60)
h_muon_dedx7 = r.TH1F("mdedx7", "Muon DeDx Layer 7", 120, 0, 60)
h_muon_dedx8 = r.TH1F("mdedx8", "Muon DeDx Layer 8", 120, 0, 60)
h_muon_dedx9 = r.TH1F("mdedx9", "Muon DeDx Layer 9", 120, 0, 60)
h_muon_dedx10 = r.TH1F("mdedx10", "Muon DeDx Layer 10", 120, 0, 60)
h_muon_dedx11 = r.TH1F("mdedx11", "Muon DeDx Layer 11", 120, 0, 60)
h_muon_dedx12 = r.TH1F("mdedx12", "Muon DeDx Layer 12", 120, 0, 60)
h_muon_dedx13 = r.TH1F("mdedx13", "Muon DeDx Layer 13", 120, 0, 60)
h_muon_dedx14 = r.TH1F("mdedx14", "Muon DeDx Layer 14", 120, 0, 60)
h_muon_dedx15 = r.TH1F("mdedx15", "Muon DeDx Layer 15", 120, 0, 60)
h_muon_dedx16 = r.TH1F("mdedx16", "Muon DeDx Layer 16", 120, 0, 60)

h_muon_dedx = [h_muon_dedx1, h_muon_dedx2, h_muon_dedx3,h_muon_dedx4, h_muon_dedx5, h_muon_dedx6, h_muon_dedx7, h_muon_dedx8, h_muon_dedx9, h_muon_dedx10, h_muon_dedx11, h_muon_dedx12, h_muon_dedx13, h_muon_dedx14, h_muon_dedx15, h_muon_dedx16]

h_fake_maxdedx = r.TH1F("maxdedx", "Max Fake DeDx", 16, 0, 16)
h_fake_mindedx = r.TH1F("mindedx", "Min Fake DeDx", 16, 0, 16)
h_muon_maxdedx = r.TH1F("mumaxdedx", "Max Muon DeDx", 16, 0, 16)
h_muon_mindedx = r.TH1F("mumindedx", "Min Muon DeDx", 16, 0, 16)

h_fake_minmax = r.TH1F("fakeMinMax", "Max-Min Fake DeDx", 200, 0, 50)
h_muon_minmax = r.TH1F("muonMinMax", "Max-Min Muon DeDx", 200, 0, 50)

h_muon_hit1 = r.TH1F("muonHit1", "1st Hit DeDx Muon", 120, 0, 60)
h_muon_hit2 = r.TH1F("muonHit2", "2nd Hit DeDx Muon", 120, 0, 60)
h_muon_hit3 = r.TH1F("muonHit3", "3rd Hit DeDx Muon", 120, 0, 60)
h_muon_hit4 = r.TH1F("muonHit4", "4th Hit DeDx Muon", 120, 0, 60)
h_muon_hit5 = r.TH1F("muonHit5", "5th+ Hits DeDx Muon", 120, 0, 60)

h_muonHits = [h_muon_hit1, h_muon_hit2, h_muon_hit3, h_muon_hit3, h_muon_hit4, h_muon_hit5]

h_fake_hit1 = r.TH1F("fakeHit1", "1st Hit DeDx Fake", 120, 0, 60)
h_fake_hit2 = r.TH1F("fakeHit2", "2nd Hit DeDx Fake", 120, 0, 60)
h_fake_hit3 = r.TH1F("fakeHit3", "3rd Hit DeDx Fake", 120, 0, 60)
h_fake_hit4 = r.TH1F("fakeHit4", "4th Hit DeDx Fake", 120, 0, 60)
h_fake_hit5 = r.TH1F("fakeHit5", "5th+ Hits DeDx Fake", 120, 0, 60)

h_fakeHits = [h_fake_hit1, h_fake_hit2, h_fake_hit3, h_fake_hit4, h_fake_hit5]

n_mark = [0]
n_mark = np.array(n_mark)
n_mark = n_mark.astype("float64")
center_mark = r.TGraph(1, n_mark, n_mark)

c1 = r.TCanvas("canvas1", "canvas1", 800, 600)
c2 = r.TCanvas("calPlots", "Calorimeter and MS Images", 800, 600)
c2.Divide(2,2)
c3 = r.TCanvas("trackerPlots", "Tracker Images", 1000, 1000)
c3.Divide(4,4)

myChain = r.TChain('trackImageProducer/tree')

for filename in os.listdir(imageDir):
    if filename.endswith(".root"):
        myChain.Add(imageDir + filename)

for iEvent, event in enumerate(myChain):
    if(iEvent%1000==0): print(iEvent)
    #if(iEvent > 10000): break
    nPV = event.nPV

    for iTrack,track in enumerate(event.tracks):
        min_dR = 10e7
        gen_matchedID = 0
        gen_matchedPT = -1
        if track.eta > 2.4: continue
        if track.inGap: continue
        if track.nValidHits < 4: continue
        if track.nValidPixelHits < 4: continue
        if track.missingMiddleHits > 0: continue
        #if track.missingOuterHits > 3: continue
        if track.missingInnerHits > 0: continue

        for iGen,genParticle in enumerate(event.genParticles):
            dR = np.sqrt(abs(track.eta - genParticle.eta)**2 + abs(track.phi - genParticle.phi)**2)
            if dR < min_dR: 
                min_dR = dR
                gen_matchedID = genParticle.pdgId
                gen_matchedPT = genParticle.pt
        hist_dR.Fill(min_dR)
        
        #Section for Muons
        if min_dR <= dR_cut:
            if not abs(gen_matchedID) == 13: continue
            if gen_matchedPT < 55: continue
            muonCount += 1
            h_d0Other.Fill(track.d0)
            h_dZOther.Fill(track.dz)
        
            maxDeDx = -10e4
            minDeDx = 10e4
            hit_layers = []
            hit_charges = []
            for iDeDx, DeDx in enumerate(track.dEdxInfo):
                if(DeDx.hitPosX == -50 or DeDx.hitPosY == -50): continue
                if(DeDx.hitLayerId == -10): continue

                if(DeDx.charge > maxDeDx): maxDeDx = DeDx.charge
                if(DeDx.charge < minDeDx): minDeDx = DeDx.charge

                #eta = track.eta
                #phi = track.phi
                xpos = DeDx.hitPosX
                ypos = DeDx.hitPosY
                layer = DeDx.hitLayerId
                hit_layers.append(layer)
                hit_charges.append(DeDx.charge)

                if(layer > 15 or layer < 0): continue

                if(OverUnderFlow):
                    if(ypos >= trkY_ub): xpos = trkY_ub
                    if(ypos <= trkY_lb): xpos = trkY_lb
                    if(xpos >= trkX_ub): xpos = trkX_ub
                    if(xpos <= trkX_lb): xpos = trkX_lb
                else:
                    if(ypos > trkY_ub or ypos < trkY_lb): continue
                    if(xpos > trkX_ub or xpos < trkX_lb): continue

                h_MuonTrk[layer].Fill(xpos, ypos)
                h_muon_dedx[layer].Fill(DeDx.charge)
            max_layers, max_charges = getMaxLayerHits(hit_layers, hit_charges)
            for i in range(len(max_layers)):
                if i < 5: h_muonHits[i].Fill(max_charges[i])
                else: h_muonHits[4].Fill(max_charges[i])

            h_muon_minmax.Fill(maxDeDx-minDeDx)
            h_muon_maxdedx.Fill(maxDeDx)
            h_muon_mindedx.Fill(minDeDx)



        #Section for Fakes
        if min_dR > dR_cut:

            largeY = False

            h_d0Fake.Fill(track.d0)
            h_dZFake.Fill(track.dz)
            fakeCount += 1
            
            for iHit, hit in enumerate(event.recHits):
                dEta = track.eta - hit.eta
                dPhi = track.phi - hit.phi
                if abs(dPhi) > math.pi:
                    dPhi -= round(dPhi / (2. * math.pi)) * 2. * math.pi

                if(OverUnderFlow):
                    if(dPhi > phi_ub): dPhi = phi_ub
                    if(dPhi < phi_lb): dPhi = phi_lb
                    if(dEta > eta_ub): dEta = eta_ub
                    if(dEta < eta_lb): dEta = eta_lb
                else:
                    if(dPhi > phi_ub or dPhi < phi_lb): continue
                    if(dEta > eta_ub or dEta < eta_lb): continue

                channel = type_to_channel(hit.detType)
                #print("Fake Candidate: ", fakeCount, "eta: ", dEta, "phi:", dPhi, "channel:", channel, "energy:", hit.energy)
                if(channel == -1): continue

                if channel != 3: 
                    h_Calo[channel].Fill(dEta, dPhi, hit.energy)
                else: 
                    h_Calo[3].Fill(dEta, dPhi)

            maxDeDx = -10e4
            minDeDx = 10e4
            hit_layers = []
            hit_charges = []
            for iDeDx, DeDx in enumerate(track.dEdxInfo):
                if(DeDx.hitPosX == -50 or DeDx.hitPosY == -50): continue
                if(DeDx.hitLayerId == -10): continue
          
                if(DeDx.charge > maxDeDx): maxDeDx = DeDx.charge
                if(DeDx.charge < minDeDx): minDeDx = DeDx.charge

                #eta = track.eta
                #phi = track.phi
                xpos = DeDx.hitPosX
                ypos = DeDx.hitPosY
                layer = DeDx.hitLayerId
                hit_layers.append(layer)
                hit_charges.append(DeDx.charge)
              
                if(layer > 15 or layer < 0): continue
 
                if(OverUnderFlow):
                    if(ypos >= trkY_ub): xpos = trkY_ub
                    if(ypos <= trkY_lb): xpos = trkY_lb
                    if(xpos >= trkX_ub): xpos = trkX_ub
                    if(xpos <= trkX_lb): xpos = trkX_lb
                else: 
                    if(ypos > trkY_ub or ypos < trkY_lb): continue
                    if(xpos > trkX_ub or xpos < trkX_lb): continue

                # boolean to decide if event should be saved for fireworks viewing          
                if(ypos > 20): largeY = True

                h_Tracker[layer].Fill(xpos, ypos)
                h_fake_dedx[layer].Fill(DeDx.charge)
            max_layers, max_charges = getMaxLayerHits(hit_layers, hit_charges)
            for i in range(len(max_layers)):
                if i < 5: h_fakeHits[i].Fill(max_charges[i])
                else: h_fakeHits[4].Fill(max_charges[i])
                
            h_fake_minmax.Fill(maxDeDx-minDeDx)
            h_fake_maxdedx.Fill(maxDeDx)
            h_fake_mindedx.Fill(minDeDx)

            if(largeY):
                # writing out run:lumi:event
                event_list.write(str(event.runNumber) + ":" + str(event.lumiBlockNumber) + ":" + str(event.eventNumber) + "\n")
            
            if fakeCount < 5:    
                for i in range(4):
                    c2.cd(i+1)
                    r.gPad.SetFrameFillColor(1)
                    h_Calo[i].Draw("colz")
                    h_Calo[i].GetXaxis().SetTitle("dEta")
                    h_Calo[i].GetYaxis().SetTitle("dPhi")
                    r.gStyle.SetOptStat(0000)
                    center_mark.Draw("P")
                    center_mark.SetMarkerStyle(2)
                    center_mark.SetMarkerSize(5)
                    center_mark.SetMarkerColor(0)

                c2.SaveAs(imgOutDir + "fakeTrack_calo_" + str(fakeCount) + ".png")

                for i in range(4):
                    h_Calo[i].Reset()

                #for i in range(16):
                #    c3.cd(i+1)
                #    r.gPad.SetFrameFillColor(1)
                #    h_Tracker[i].Draw("colz")
                #    h_Tracker[i].GetXaxis().SetTitle("Phi")
                #    h_Tracker[i].GetYaxis().SetTitle("Eta")
                #    r.gStyle.SetOptStat(0000)
                #    center_mark.Draw("P")
                #    center_mark.SetMarkerStyle(2)
                #    center_mark.SetMarkerSize(5)
                #    center_mark.SetMarkerColor(0)

                #c3.SaveAs(imgOutDir+ "fakeTrack_tracker_" + str(fakeCount) + ".png")

                #for i in range(4):
                #    h_Tracker[i].Reset()


print('################################################################ \n Number of Tracks with dR > '  + str(dR_cut) + ': ' + str(fakeCount) + '\n Number of Muons: ' + str(muonCount) + '\n################################################################')

hist_dR.Write()
h_d0Fake.Write()
h_d0Other.Write()
h_dZFake.Write()
h_dZOther.Write()
h_fake_minmax.Write()
h_fake_maxdedx.Write()
h_fake_mindedx.Write()
h_muon_minmax.Write()
h_muon_mindedx.Write()
h_muon_maxdedx.Write()
for i in range(16):
    if i < 5: 
        h_fakeHits[i].Write()
        h_muonHits[i].Write()
    h_Tracker[i].Write()
    h_fake_dedx[i].Write()
    h_MuonTrk[i].Write()
    h_muon_dedx[i].Write()
fout.Close()
event_list.close()












