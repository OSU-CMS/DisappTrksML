import math
from ROOT import TFile, TTree, TH2D, TCanvas, gROOT
from ROOT.Math import XYZVector

def passesIsolatedTrackSelection(track):

    momentum = XYZVector(track.px, track.py, track.pz)
    eta = momentum.Eta()
    pt = math.sqrt(momentum.Perp2())

    if not abs(eta) < 2.4: return False
    if not pt > 30: return False
    if track.inGap: return False
    if not track.nValidPixelHits >= 4: return False
    if not track.nValidHits >= 4: return False
    if not track.missingInnerHits == 0: return False
    if not track.missingMiddleHits == 0: return False
    if not track.trackIso / pt < 0.05: return False
    if not abs(track.d0) < 0.02: return False
    if not abs(track.dz) < 0.5: return False
    if not abs(track.dRMinJet) > 0.5: return False
    if not abs(track.deltaRToClosestElectron) > 0.15: return False
    # deltaRToClosestMuon
    if not abs(track.deltaRToClosestTauHad) > 0.15: return False
    return True

def printTrackInfo(track):
    momentum = XYZVector(track.px, track.py, track.pz)
    eta = momentum.Eta()
    pt = math.sqrt(momentum.Perp2())

    print '\tTrack:'

    if not abs(eta) < 2.4: print '\t\tFails eta:', eta
    if not pt > 30: print '\t\tFails pt:', pt
    if track.inGap: print '\t\tFails !inGap'
    if not track.nValidPixelHits >= 4: print '\t\tFails nValidPixelHits:', track.nValidPixelHits
    if not track.nValidHits >= 4: print '\t\tFails nValidHits:', track.nValidHits
    if not track.missingInnerHits == 0: print '\t\tFails missingInnerHits:', track.missingInnerHits
    if not track.missingMiddleHits == 0: print '\t\tFails missingMiddleHits:', track.missingMiddleHits
    if not track.trackIso / pt < 0.05: print '\t\tFails relative track iso:', track.trackIso, '/', pt, '=', track.trackIso / pt
    if not abs(track.d0) < 0.02: print '\t\tFails d0:', track.d0
    if not abs(track.dz) < 0.5: print '\t\tFails dz:', track.dz
    if not abs(track.dRMinJet) > 0.5: print '\t\tFails dRMinJet:', track.dRMinJet
    if not abs(track.deltaRToClosestElectron) > 0.15: print '\t\tFails deltaRToClosestElectron:', track.deltaRToClosestElectron
    # deltaRToClosestMuon
    if not abs(track.deltaRToClosestTauHad) > 0.15: print '\t\tFails deltaRToClosestTauHad:', track.deltaRToClosestTauHad

gROOT.SetBatch() # I am Groot.

fin = TFile('images.root')
tree = fin.Get('trackImageProducer/tree')

can = TCanvas('can', 'can', 10, 10, 800, 600)

h_ecal = TH2D('h_ecal', 'h_ecal', 50, -0.5, 0.5, 50, -0.5, 0.5)
h_preshower = TH2D('h_preshower', 'h_preshower', 50, -0.5, 0.5, 50, -0.5, 0.5)
h_hcal = TH2D('h_hcal', 'h_hcal', 50, -0.5, 0.5, 50, -0.5, 0.5)
h_muon = TH2D('h_muon', 'h_muon', 50, -0.5, 0.5, 50, -0.5, 0.5)

nEvents = tree.GetEntries()
nTagMuons = 0
nPassingProbes = 0

foundFirst = False

for event in tree:

    nTags_thisEvent = 0
    nProbes_thisEvent = 0
    nTracks_thisEvent = len(event.tracks)

    for track in event.tracks:

        if passesIsolatedTrackSelection(track):
            if abs(track.deltaRToClosestMuon) > 0.15:
                nPassingProbes += 1
                nProbes_thisEvent += 1
            else:
                nTagMuons += 1
                nTags_thisEvent += 1
                continue

        if foundFirst:
            continue

        momentum = XYZVector(track.px, track.py, track.pz)
        eta = momentum.Eta()
        phi = momentum.Phi()

        for hit in event.recHits:
            dEta = eta - hit.eta
            dPhi = phi - hit.phi            
            # branch cut [-pi, pi)
            if abs(dPhi) > math.pi:
                dPhi -= round(dPhi / (2. * math.pi)) * 2. * math.pi

            if hit.detType == 1 or hit.detType == 2:
                h_ecal.Fill(dEta, dPhi, hit.energy)
            elif hit.detType == 3:
                h_preshower.Fill(dEta, dPhi, hit.energy)
            elif hit.detType == 4:
                h_hcal.Fill(dEta, dPhi, hit.energy)
            elif hit.detType > 4:
                h_muon.Fill(dEta, dPhi) # no energy, just 1

        foundFirst = True

    print 'this event:', nTags_thisEvent, 'tags,', nProbes_thisEvent, 'probes', '(total tracks:', nTracks_thisEvent, ')'

    if nProbes_thisEvent == 0:
        for track in event.tracks: printTrackInfo(track)

print 'In', nEvents, 'events, found:'
print nTagMuons, 'tag muons'
print nPassingProbes, 'passing probes'

h_ecal.Draw('colz')
can.SaveAs('ecal.png')

h_preshower.Draw('colz')
can.SaveAs('preshower.png')

h_hcal.Draw('colz')
can.SaveAs('hcal.png')

h_muon.Draw('colz')
can.SaveAs('muon.png')
