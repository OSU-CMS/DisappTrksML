import numpy as np
import os, sys
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
import pickle

indata = np.load("train_data.npy",allow_pickle=True)
indata2 = np.load("singleMu_forTraining.npy",allow_pickle=True)
X = [None, None]
# X[0] = indata[1][:,:10,:]			# MC muon failures		
# X[0] = indata2[0][:,:10,:]		# higgsino
X[0] = indata[3][:1000,:10,:]		# MC bkg
X[1] = indata2[0][:,:10,:]			# TP muon failures
class0 = np.reshape(X[0], (len(X[0]), X[0].shape[1]*X[0].shape[2]))
class1 = np.reshape(X[1], (len(X[1]), X[1].shape[1]*X[1].shape[2]))
X = np.vstack((class0, class1))
y = np.concatenate((np.zeros(len(class0)), np.ones(len(class1))))
X, y = shuffle(X, y, random_state=42)
# clf = make_pipeline(MinMaxScaler(), LinearSVC(random_state=42, tol=1e-6))
# clf = make_pipeline(MinMaxScaler(), 
# 					GradientBoostingClassifier(n_estimators=500, learning_rate=1.0,
#  					max_depth=10, random_state=42))
nTotal = len(class0) + len(class1)
f0 = len(class0)*1.0/nTotal
f1 = len(class1)*1.0/nTotal
# class_prior = [f0, f1]
clf = make_pipeline(MinMaxScaler(), ComplementNB())

clf.fit(X, y)
pickle.dump(clf, open("rfclf.sav", 'wb'))

clf =pickle.load(open("rfclf.sav", 'rb'))
indata = np.load("test.npy",allow_pickle=True)
X_val = []
classLabels = ["SingleMuon", "Higgsino 700GeV 10cm",
				"Higgsino 700GeV 100cm", "Higgsino 700GeV 1000cm",
				"Higgsino 700GeV 1000cm"]
for i, iClass in enumerate(indata):
	iClass = iClass[:,:10,:]
	iClass = np.reshape(iClass, (len(iClass), iClass.shape[1]*iClass.shape[2]))
	preds = clf.predict(iClass)
	nMuons = np.sum(preds)*1.0
	nBkg = len(preds) - nMuons
	print classLabels[i]
	print "\tLabelled as muons: {0}".format(nMuons)
	print "\tLabelled as non-muons: {0}".format(nBkg)