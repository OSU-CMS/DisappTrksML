# Pipeline

## 1. Data production
fill in

## 2. Selection on root files
From the root files produced in Step 1, apply a selection using **selection/makeSelection.cpp**. Once the selections are applied, each file will contain two trees, electrons (eTree) and backgrounds (bTree) determined by gen-matching.

## 3. Convert data to feed to classifiers
From the root files produced in Step 2 (or Step 1), **convert/** includes files to convert the rec hits around each track to some representation that can then be fed to classifiers.
- **toImages.py** produces 4+(resolution_eta,resolution_phi,4) images, of the
(ECAL, HS, HCAL, MUO) systems.
- **toSets.py** produces 4+(number_of_hits,(dEta,dPhi,energy,detIndex).

The "4+" in both represents [eventNumber, lumiBlockNumber, runNumber, trackNumber], which is included in every track. Each file produces an "infos" array in each track:

infos =
  1. eventNumber, 
  2. lumiBlockNumber, 
  3. runNumber,
  4. trackNumber
  5. class_label,
  6. nPV,
  7. track.deltaRToClosestElectron,
  8. track.deltaRToClosestMuon,
  9. track.deltaRToClosestTauHad,
  10. track_eta,
  11. track_phi,
  12. track.dRMinBadEcalChannel
  13. track.nLayersWithMeasurement
