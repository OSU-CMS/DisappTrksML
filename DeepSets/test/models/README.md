The **electron models** are the ones picked from the kfold validation outlined in the ELOG from March 12, 2021.
The K19 model was the one used to produce the plots and tables in the ELOG from April 30, 2021. This model was trained using 50% of each training batch as electrons, the highest undersampling tested.
These models were trained using Monte Carlo events that failed the electron reconstruction and the 'trainingSelection' from the pyhon/ElectronModel.py, and the task
was to classify electrons (gen-matched tracks) and non electrons (tracks far from generated electrons).
Each track used the 100 most energetic hits in the ECAL, HCAL, and MUO, with each hit being of form (dPhi, dEta, detType, Energy), and track-level information was added as track's absolute eta and phi coordinates, the number of hits in the tracker associated to that track, and the amount of pileup in the event.

The **moun models** were _not_ optimized using a systmatic kfold validation (lack of time), different models were tested and their validation peformance evluated, without
kfold validation; this model was the best profming out of the ones tested. The ELOG from April 30, 2021 outlines the performance of this model.
These models were trained using Monte Carlo events, classifying between muon tracks that were reconstructed as muons and tracks from non-muon events, both passing the 'trainingSelection' from python/MuonModel.py.
(_Why train on reconstructed muons, if the goal is to identify muon tracks that failed the reconstruction?_
Because there were too few muons that failed the reconstruction to train on. The assumption, which is tested through this model, is that training on 'good' muons
will also teach the model to identify non-reconstructed muons).
Each track used the 20 nearest MUO hits, with each hit being of form (dPhi, dEta, Station Number, Time, detType on-hot encoded), and track-level information was added as track's absolute eta and phi coordinates, the amount of pileup, the sum of the energy deposited in the ECAL and HCAL (separately) within delta R < 0.1 of the track, the pT error, and the reduced chi squared.

See ELOG and Luca's senior thesis for more details.
