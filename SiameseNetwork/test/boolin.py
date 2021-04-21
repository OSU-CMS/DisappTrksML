import numpy as np
import os, sys
import glob
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB


# load a list of files
def load_from_files(files, dirname='', obj='tracks'):
	X = None
	for file in files:
		infile = np.load(dirname + file, allow_pickle=True)[obj]
		if len(infile) == 0: continue
		if X is None: X = infile
		else: X = np.vstack((X, infile))
	return X

# load training data
dataDir = "/store/user/llavezzo/disappearingTracks/naiveBayes/"
trainFiles = glob.glob(dataDir+"SingleMu_training_pt1/*.root.npz")
X_train = load_from_files(trainFiles, obj='signal_tracks')
X_train_bkg = load_from_files(trainFiles, obj='bkg_trakcs')

print X_train.shape
print X_train_bkg.shape

# combine and shuffle train data
class0 = np.reshape(X_train_bkg, (len(X_train_bkg), X_train_bkg.shape[1]*X_train_bkg.shape[2]))
class1 = np.reshape(X_train, (len(X_train), X_train.shape[1]*X_train.shape[2]))
X = np.vstack((class0, class1))
y = np.concatenate((np.zeros(len(class0)), np.ones(len(class1))))
X, y = shuffle(X, y, random_state=42)

# fit
# clf = make_pipeline(MinMaxScaler(), RandomForestClassifier())
clf = RandomForestClassifier(max_depth=20)
clf.fit(X, y)
pickle.dump(clf, open("clf.sav", 'wb'))

# training performance (on training data)
preds_class0 = clf.predict(X[y == 0])
preds_class1 = clf.predict(X[y == 1])
nMuons = np.sum(preds_class0)*1.0
nBkg = len(preds_class0) - nMuons
print "Train Background"
print "\tLabelled as muons: {0}".format(nMuons)
print "\tLabelled as non-muons: {0}".format(nBkg)
nMuons = np.sum(preds_class1)*1.0
nBkg = len(preds_class1) - nMuons
print "Train Muons"
print "\tLabelled as muons: {0}".format(nMuons)
print "\tLabelled as non-muons: {0}".format(nBkg)

# validate
dataDir = "/store/user/llavezzo/disappearingTracks/muonsTesting/"
clf =pickle.load(open("clf.sav", 'rb'))

# directories to validate
valDirs = ["SingleMuon_fullSel_pt1_FIXED", "SingleMuon_fullSel_pt2_FIXED",
			"higgsino_700GeV_10cm", "higgsino_700GeV_100cm",
			"higgsino_700GeV_1000cm", "higgsino_700GeV_10000cm"]
for valDir in valDirs:
	if "higgsino" in valDir: obj = 'signal'
	else: obj = 'tracks'
	X_val = load_from_files(glob.glob(dataDir+valDir+"/*.root.npz"), obj=obj)
	X_val = np.reshape(X_val, (len(X_val), X_val.shape[1]*X_val.shape[2]))
	preds = clf.predict(X_val)

	# # debugging??
	# preds = clf.predict_proba(X_val)
	# plt.hist(preds[:,1], label=valDir)

	nMuons = np.sum(preds)*1.0
	nBkg = len(preds) - nMuons
	print valDir
	print "\tLabelled as muons: {0}".format(nMuons)
	print "\tLabelled as non-muons: {0}".format(nBkg)

# debugging
plt.legend()
plt.savefig("test.png")