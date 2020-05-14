from ROOT import TFile, TTree, TH2D, TCanvas, gROOT

gROOT.SetBatch() # I am Groot.

fin = TFile('images.root')
tree = fin.Get('trackImageProducer/tree')

can = TCanvas('can', 'can', 10, 10, 800, 600)

h = TH2D('h', 'h', 50, -0.5, 0.5, 50, -0.5, 0.5)

for iTrack, track in enumerate(tree):
        if abs(track.genMatchedDR) >= 0.05 or abs(track.genMatchedID) != 11: continue
        for i in range(len(track.recHits_dEta)):
                if track.recHits_detType != 1:
                        h.Fill(track.recHits_dEta[i], track.recHits_dPhi[i], track.recHits_energy[i])

        h.Draw('colz')
        can.SaveAs('ecal.png')
        break