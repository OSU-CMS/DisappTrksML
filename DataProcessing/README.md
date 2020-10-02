# Pipeline:

## 1. Data production:
fill in

## 2. Selection on root files:
From the root files produced in Step 1, apply a selection using **selection/makeSelection.cpp**. Once the selections are applied, each file will contain two trees, electrons (eTree) and backgrounds (bTree) determined by gen-matching.

## 3. Convert data to feed to classifiers:
From the root files produced in Step 2 (or Step 1), **convert/** includes files to convert the rec hits around each track to some representation that can then be fed to classifiers.
- **toImages.py** produces 3+(resolution_eta,resolution_phi,4) images, of the
(ECAL, HS, HCAL, MUO) systems.
- **toSets.py** produces 3+(number_of_hits,(dEta,dPhi,energy,detIndex).

The "3+" in both represents [eventNumber, lumiBlockNumber, runNumber], which is included in every event. Each file produces an "infos" array in each event:

infos =
  1. eventNumber, 
  2. lumiBlockNumber, 
  3. runNumber,
  4. class_label,
  5. nPV,
  6. track.deltaRToClosestElectron,
  7. track.deltaRToClosestMuon,
  8. track.deltaRToClosestTauHad,
  9. track_eta,
  10. track_phi,
  11. track.dRMinBadEcalChannel
