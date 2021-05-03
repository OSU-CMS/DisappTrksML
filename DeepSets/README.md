# Workflow #
Big picture: `architecture.py` contains a general 'Architecture' class to apply selections and convert `.root` files into `.npy/.npz` files that can be taken as input to the deepSets model.
`ElectronModel.py, MuonModel.py` are the specific model architectures for each the different lepton classification tasks.
<br />
<br />
**Convert data** using `convertToNumpy.py`; in here, you must initialize your model architecture with the phi/eta window range and the maximum number of hits to be considered for each track.
Then, each `LeptonModel.py` has different `convertBLANKFileToNumpy` functions which are to be used here to convert whichever type of file (Monte Carlo, Tag and Probe, AMSB, etc.)
to npy. Within each function, you can decide what selection to apply: these selections are available in either `architecture.py` or each `LeptonModel.py`.
`run_convert.py` and the wrapper then provide a way to autmate the conversion of task over a whole directory using HTCondor, once `convertToNumpy.py` is set up as desired.
<br />
<br />
**Train the models** using `train.py`; in here, you must initialize your model architecture again, this time it's important to specify the rho and phi network layers.
Also, you might specify what track-level information to train on using the `track_info_indices` array, which refer to the indices of the `infos` output of the `convertTrackFromTree`
of each LeptonModel.py from the previous step.
The model will output the training history, weights, and metrics to a folder.
`run_train.py` and the wrapper also provide a way to submit the training to an HTCondor job.
<br />
<br />
**Evaluate the models** using the `evaluateLeptonModel.py` scripts. You need to initiliaze your architecture, load the weights you have saved, and then evaluate the model.
