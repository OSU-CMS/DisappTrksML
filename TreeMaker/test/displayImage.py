import math
from ROOT import TFile, TTree, TH2D, TCanvas, gROOT

gROOT.SetBatch() # I am Groot.

fin = TFile('images.root')
tree = fin.Get('trackImageProducer/tree')

can = TCanvas('can', 'can', 10, 10, 800, 600)

h_ecal = TH2D('h_ecal', 'h_ecal', 50, -0.5, 0.5, 50, -0.5, 0.5)
h_preshower = TH2D('h_preshower', 'h_preshower', 50, -0.5, 0.5, 50, -0.5, 0.5)
h_hcal = TH2D('h_hcal', 'h_hcal', 50, -0.5, 0.5, 50, -0.5, 0.5)
h_muon = TH2D('h_muon', 'h_muon', 50, -0.5, 0.5, 50, -0.5, 0.5)

goodTrack = False

for event in tree:

    for iTrack in range(len(event.track_eta)):
        if abs(event.track_genMatchedDR[iTrack]) >= 0.05: continue
        if abs(event.track_genMatchedID[iTrack]) != 13: continue
        if event.track_genMatchedPt[iTrack] < 35: continue

        goodTrack = True

        for iHit in range(len(event.recHits_eta)):
            dEta = event.track_eta[iTrack] - event.recHits_eta[iHit]
            dPhi = event.track_phi[iTrack] - event.recHits_phi[iHit]            
            # branch cut [-pi, pi)
            if abs(dPhi) > math.pi:
                dPhi -= round(dPhi / (2. * math.pi)) * 2. * math.pi

            if event.recHits_detType[iHit] == 1 or event.recHits_detType[iHit] == 2:
                h_ecal.Fill(dEta, dPhi, event.recHits_energy[iHit])
            elif event.recHits_detType[iHit] == 3:
                h_preshower.Fill(dEta, dPhi, event.recHits_energy[iHit])
            elif event.recHits_detType[iHit] == 4:
                h_hcal.Fill(dEta, dPhi, event.recHits_energy[iHit])
            elif event.recHits_detType[iHit] > 4:
                h_muon.Fill(dEta, dPhi) # no energy, just 1

        break
    if goodTrack: break

h_ecal.Draw('colz')
can.SaveAs('ecal.png')

h_preshower.Draw('colz')
can.SaveAs('preshower.png')

h_hcal.Draw('colz')
can.SaveAs('hcal.png')

h_muon.Draw('colz')
can.SaveAs('muon.png')
