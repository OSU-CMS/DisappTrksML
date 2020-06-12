from xgboost import XGBClassifier
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from collections import Counter
from utils import *

dataDir = '/data/disappearingTracks/'
workDir = '/home/llavezzo/'
plotDir = workDir + 'plots/xgboost/'

os.system('mkdir '+str(plotDir))

#config parameters
fname = 'images_DYJets50_norm_40x40.pkl'
pos_class = [1]
neg_class = [0,2]

# extract data and classes
df = pd.read_pickle(dataDir+fname)
df_recofail = df.loc[df['deltaRToClosestElectron']>0.15]
x = df_recofail.iloc[:,4:].to_numpy()
x = np.reshape(x, [x.shape[0],40,40,4])
x = x[:,:,:,[0,2,3]]
x = np.reshape(x, [x.shape[0],40*40*3])
y = df_recofail['type'].to_numpy()
for i,label in enumerate(y):
    if label in pos_class: y[i] = 1
    if label in neg_class: y[i] = 0

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

counter = Counter(y_train)
print(counter)
model = XGBClassifier(scale_pos_weight=counter[0]/counter[1])
model.fit(x_train,y_train)

print(model)

predictions = model.predict(x_test)
predictions_rounded = [round(value) for value in predictions]

print()
print("Calculating and plotting confusion matrix")
cm = calc_cm(y_test,predictions_rounded)
plot_confusion_matrix(cm,['bkg','e'],plotDir + 'cm.png')
print()

print("Plotting predictions")
plot_predictions(y_test,predictions,plotDir+'predictions.png')
print()

precision, recall = calc_binary_metrics(cm)
print("Precision = TP/(TP+FP) = fraction of predicted true actually true ",round(precision,3))
print("Recall = TP/(TP+FN) = fraction of true class predicted to be true ",round(recall,3))
auc = roc_auc_score(y_test,predictions_rounded)
print("AUC Score:",auc)
print()