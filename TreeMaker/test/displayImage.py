import math
from ROOT import TFile, TTree, TH2D, TCanvas, gROOT

gROOT.SetBatch() # I am Groot.

fin = TFile('images.root')
tree = fin.Get('trackImageProducer/tree')

can = TCanvas('can', 'can', 10, 10, 800, 600)

h = TH2D('h', 'h', 50, -0.5, 0.5, 50, -0.5, 0.5)

goodTrack = False

for event in tree:

    for iTrack in range(len(event.track_eta)):
        if abs(event.track_genMatchedDR[iTrack]) >= 0.05: continue
        if abs(event.track_genMatchedID[iTrack]) != 11: continue
        if event.track_genMatchedPt[iTrack] < 35: continue

        goodTrack = True

        for iHit in range(len(event.recHits_eta)):
            if event.recHits_detType[iHit] != 1 and event.recHits_detType[iHit] != 2: continue
            dEta = event.track_eta[iTrack] - event.recHits_eta[iHit]
            dPhi = event.track_phi[iTrack] - event.recHits_phi[iHit]
            # branch cut [-pi, pi)
            if abs(dPhi) > math.pi:
                dPhi -= round(dPhi / (2. * math.pi)) * 2. * math.pi
            h.Fill(dEta, dPhi, event.recHits_energy[iHit])

        break
    if goodTrack: break

h.Draw('colz')
can.SaveAs('ecal.png')
