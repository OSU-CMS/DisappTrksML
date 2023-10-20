import warnings

warnings.filterwarnings("ignore")
import glob, os

import tensorflow as tf
import cmsml

from DisappTrksML.DeepSets.ElectronModel import *
from DisappTrksML.DeepSets.MuonModel import *
from DisappTrksML.DeepSets.generator import *
from DisappTrksML.DeepSets.utilities import *

if False:
    # limit CPU usage
    config = tf.compat.v1.ConfigProto(
        inter_op_parallelism_threads=4,
        intra_op_parallelism_threads=4,
        allow_soft_placement=True,
        device_count={"CPU": 4},
    )
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

#######

backup_suffix = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
outdir = "train_increase_phi_neurons/"


info_indices = [4, 8, 9, 12]  # nPV  # eta  # phi  # nValidPixelHits
model_params = {
    "eta_range": 0.25,
    "phi_range": 0.25,
    "phi_layers": [128, 128, 512],
    "f_layers": [64, 64, 64],
    "max_hits": 100,
    "track_info_indices": info_indices,
}
val_generator_params = {
    "input_dir": "/store/user/rsantos/2022/combined_DYJet/training/",
    "batch_size": 256,
    "max_hits": 100,
    "info_indices": info_indices,
}
train_generator_params = val_generator_params.copy()
train_generator_params.update({"shuffle": True, "batch_ratio": 0.5})

if len(sys.argv) > 1:
    input_params = np.load("params.npy", allow_pickle=True)[int(sys.argv[1])]
    model_params["phi_layers"] = input_params[0]
    model_params["f_layers"] = input_params[1]
    outdir = str(input_params[-1])



train_params = {"epochs": 100, "outdir": outdir, "patience_count": 20}

print("output directory", outdir)
if not os.path.isdir(outdir):
    os.mkdir(outdir)

arch = ElectronModel(**model_params)
arch.buildModel()

inputFiles = glob.glob(train_generator_params["input_dir"] + "images_*.root.npz")
inputIndices = np.array([f.split("images_")[-1][:-9] for f in inputFiles])
nFiles = len(inputIndices)
print(("Found", nFiles, "input files"))

# Use a 70 20 10 split for training, validation, and testing
file_ids = {
    "train": inputIndices[: int(0.8 * nFiles)],
    "validation": inputIndices[int(0.8 * nFiles) :],
}

train_generator = BalancedGenerator(file_ids["train"], **train_generator_params)
val_generator = Generator(file_ids["validation"], **val_generator_params)

arch.fit_generator(
    train_generator=train_generator, val_generator=val_generator, **train_params
)

arch.saveGraph()

#arch.save_trainingHistory(train_params["outdir"] + "trainingHistory.pkl")
#arch.plot_trainingHistory(
#    train_params["outdir"] + "trainingHistory.pkl",
#    train_params["outdir"] + "trainingHistory.png",
#    "loss",
#)

arch.save_weights(train_params["outdir"] + "model_weights.h5")
arch.save_model(train_params["outdir"] + "model.h5")
#arch.save_metrics(
#    train_params["outdir"] + "trainingHistory.pkl",
#    train_params["outdir"] + "metrics.pkl",
#    train_params,
#)
