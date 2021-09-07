import numpy as np
import os, sys
import glob
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from DisappTrksML.MuonSklearnClassifiers.architecture import *

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

classifiers = [
	make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors = 1)),
	make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors = 5)),
	make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors = 10)),
	make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators = 100, max_depth=10)),
	make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators = 100, max_depth=20)),
	make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators = 100, max_depth=50)),
	make_pipeline(MinMaxScaler(), DecisionTreeRegressor()),
	make_pipeline(MinMaxScaler(), GradientBoostingClassifier()),
	make_pipeline(MinMaxScaler(), GaussianNB()),
	make_pipeline(MinMaxScaler(), MultinomialNB()),
	make_pipeline(MinMaxScaler(), ComplementNB()),
	make_pipeline(MinMaxScaler(), LinearSVC(random_state=0))
]

labels = [
	"KNeighborsClassifier",
	"KNeighborsClassifier5",
	"KNeighborsClassifier10",
	"RandomForestClassifier1",
	"RandomForestClassifier2",
	"RandomForestClassifier3",
	"DecisionTreeRegressor",
	"GradientBoostingClassifier",
	"GaussianNB",
	"MultinomialNB",
	"ComplementNB",
	"LinearSVC"
]

assert len(labels) == len(classifiers)

train_pre, train_rec = [], []
val_pre, val_rec = [], []

for clf, label in zip(classifiers, labels):

	print
	print label

	arch = GeneralArchitecture(model = clf)

	# load training data
	dataDir = "/store/user/llavezzo/disappearingTracks/muonsTesting/"
	trainFiles = glob.glob(dataDir+"SingleMu_training_pt2/*.root.npz")
	X_train = arch.load_from_files(trainFiles, obj='signal_tracks')
	X_train_bkg = arch.load_from_files(trainFiles, obj='bkg_trakcs')[:500,:,:]

	print X_train.shape
	print X_train_bkg.shape

	# combine and shuffle train data
	class0 = np.reshape(X_train_bkg, (len(X_train_bkg), X_train_bkg.shape[1]*X_train_bkg.shape[2]))
	class1 = np.reshape(X_train, (len(X_train), X_train.shape[1]*X_train.shape[2]))
	X = np.vstack((class0, class1))
	y = np.concatenate((np.zeros(len(class0)), np.ones(len(class1))))
	X, y = shuffle(X, y, random_state=42)

	# fit
	arch.model.fit(X, y)
	pickle.dump(arch.model, open("clf.sav", 'wb'))

	# training performance (on training data)
	preds = arch.model.predict(X)
	cm = arch.calc_binary_cm(y, preds)
	precision, recall, f1 = arch.calc_binary_metrics(cm)
	print "Training performance"
	print cm
	print "Precision", round(precision,2), "Recall", round(recall,2), "F1", round(f1,2)
	train_pre.append(precision)
	train_rec.append(recall)

	# evaluation on validation data with training selection applied
	valFiles = glob.glob(dataDir+"SingleMu_training_pt1/*.root.npz")
	X_val_signal = arch.load_from_files(valFiles, obj='signal_tracks')
	X_val_bkg = arch.load_from_files(valFiles, obj='bkg_trakcs')
	X_val = np.vstack((X_val_signal, X_val_bkg))
	X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1]*X_val.shape[2]))
	true = np.concatenate((np.ones(X_val_signal.shape[0]), np.zeros(X_val_bkg.shape[0])))
	preds = arch.model.predict(X_val)
	cm = arch.calc_binary_cm(true, preds)
	precision, recall, f1 = arch.calc_binary_metrics(cm)
	print "Validation performance"
	print cm
	print "Precision", round(precision,2), "Recall", round(recall,2), "F1", round(f1,2)
	val_pre.append(precision)
	val_rec.append(recall)

	# validate on this data only when a good model is found
	# sys.exit(0)
	# continue

	# evaluate on data with full analysis selection applied
	valDirs = [	"SingleMuon_fullSel_pt1_FIXED",
				"higgsino_700GeV_10cm", "higgsino_700GeV_100cm",
				"higgsino_700GeV_1000cm", "higgsino_700GeV_10000cm"]
	objs = ['tracks',
			'signal', 'signal', 'signal' ,'signal']
	for valDir, obj in zip(valDirs, objs):
		X_val = arch.load_from_files(glob.glob(dataDir+valDir+"/*.root.npz"), obj=obj)
		X_val = np.reshape(X_val, (len(X_val), X_val.shape[1]*X_val.shape[2]))
		preds = arch.model.predict(X_val)
		nMuons = len(preds[preds > 0.5])
		nBkg = len(preds) - nMuons
		print valDir
		print "\tLabelled as muons: {0}".format(nMuons)
		print "\tLabelled as non-muons: {0}".format(nBkg)

	del arch

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1,2,1)
ax_b = fig.add_subplot(1,2,2)
ax.set_ylim(0,1.2)
ax_b.set_ylim(0,1.2)
x = np.arange(len(labels))
ax.scatter(x, train_pre, marker="x", color="blue", label="Train Precision")
ax.scatter(x, val_pre, marker="x", color="red", label="Validation Precision")
ax_b.scatter(x, train_rec, marker="+", color="blue", label="Train Recall")
ax_b.scatter(x, val_rec, marker="+", color="red", label="Validation Recall")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=90)
ax.legend()
ax_b.set_xticks(x)
ax_b.set_xticklabels(labels, rotation=90)
ax_b.legend()
fig.tight_layout()
fig.savefig("classifier_comparision.png", dpi=200)
