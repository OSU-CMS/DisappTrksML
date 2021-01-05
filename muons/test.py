import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

dataDir = '/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_muons/'
data = None
cnt = 0
for file in os.listdir(dataDir):
	indata = np.load(dataDir+file,allow_pickle=True)['background']
	if data is None: data = indata
	else: data = np.vstack((data,indata))

	cnt+=1
	if cnt > 100: break
data = np.reshape(data,(len(data),400))
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

signal = None
for file in os.listdir(dataDir):
	indata = np.load(dataDir+file,allow_pickle=True)['signal']
	if len(indata) == 0: continue
	if signal is None: signal = indata
	else: signal = np.vstack((signal,indata))
	if len(signal) > 100: break
signal = np.reshape(signal,(len(signal),400))

print("Imported",len(signal),"muons")
print("Imported",len(data),"background")

clf = IsolationForest(n_estimators=500, n_jobs=-1, contamination=0, random_state=42)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_signal = clf.predict(signal)

print("Train",len(X_train),"outliers",len(y_pred_train[y_pred_train>0]))
print("Test",len(X_test),"outliers",len(y_pred_test[y_pred_test>0]))
print("Signal",len(signal),"outliers",len(y_pred_signal[y_pred_signal>0]))