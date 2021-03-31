import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys

from DisappTrksML.SiameseNetwork.SiameseNetwork import *

# build model from architecture
arch = SiameseNetwork(phi_layers = [64,32], f_layers=[32],
				eta_range=1.0,phi_range=1.0,max_hits=20)
arch.load_model("models/siam_model.1000.h5")

# load in the data
X_val = arch.load_data("val_data.npy", nEvents = 25, nClasses=2)
X_val_ref = arch.load_data("ref_data.npy", nEvents = 30, nClasses=2)
arch.add_data(X_val=X_val, X_val_ref=X_val_ref)

# number of events in reference set
N = X_val_ref.shape[1]

n_correct = [0,0]

for iClass, events in enumerate(X_val):

    for event in events:

        # test one event at a time against the reference set
        test_images = np.asarray([event]*N)

        # similarity score with other class
        other_class_preds = arch.model.predict([test_images, X_val_ref[(iClass + 1)%2, :, :, :]])

        # similiarity score with same class
        same_class_preds = arch.model.predict([test_images,  X_val_ref[iClass, :, :, :]])

        # higher similiarity score wins
        print(np.mean(same_class_preds), np.mean(other_class_preds))
        if np.mean(same_class_preds) > np.mean(other_class_preds):
            n_correct[iClass] +=1 

print(n_correct)