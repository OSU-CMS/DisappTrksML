import pandas as pd
import os
import numpy as np
from sklearn.cross_validation import train_test_split
import json

dataDirIn = '/store/user/llavezzo/images/'
dataDirOut = '/store/user/llavezzo/images_e_reco_failed/'

# electrons as signal, bkg everything else
pos_class = [1]
neg_class = [0,2]

ids = []
labels = []
batch_size = 1000

batchNum = 0
partition_labels = {}

for filename in os.listdir(dataDirIn):	
	if(not filename.endswith('.pkl')): continue

	dftemp = pd.read_pickle(dataDirIn+filename)
	
	# more efficient
	dftemp = dftemp.astype('float32')
	dftemp['type'] = dftemp['type'].astype('int32')

	# select only electron RECO failed events
	dftemp = dftemp.loc[dftemp['deltaRToClosestElectron']>0.15]
	dftemp_e = dftemp.loc[dftemp['type'] == 1]
	dftemp_bkg = dftemp.loc[dftemp['type'] != 1]
	
	if(batchNum != 0): this_group_e = pd.concatenate([dftemp_e,previous_batch_e])
	else: this_group_e = dftemp_e

	if(batchNum != 0): this_group_bkg = pd.concatenate([dftemp_bkg,previous_batch_bkg])
	else: this_group_bkg = dftemp_bkg

	while(this_group_e.shape[0] >= batch_size):
		batch = this_group_e.iloc[:batch_size,:]
		batch.to_pickle(dataDirOut+'batch_'+str(batchNum)+'.pkl')
		partition_labels[batchNum] = 1
		batchNum += 1

	while(this_group_bkg.shape[0] >= batch_size):
		batch = this_group_bkg.iloc[:batch_size,:]
		batch.to_pickle(dataDirOut+'batch_'+str(batchNum)+'.pkl')
		partition_labels[batchNum] = 0
		batchNum += 1

	previous_batch_e = this_group_e
	previous_batch_bkg = this_group_bkg

# save last ones too
previous_batch_e.to_pickle(dataDirOut+'batch_'+str(batchNum)+'.pkl')
partition_labels[batchNum] = 1
batchNum+=1
previous_batch_bkg.to_pickle(dataDirOut+'batch_'+str(batchNum)+'.pkl')
partition_labels[batchNum] = 0
batchNum+=1

# partition the IDs and store the class labels per ID
ids = np.arange(0, batchNum, 1) 
train_ids, test_ids= train_test_split(ids, test_size = 0.4, random_state=42)
test_ids, valid_ids = train_test_split(test_ids, test_size=0.5,random_state=42)
partition = {'train':train_ids.tolist(),'validation':valid_ids.tolist(),'test':test_ids.tolist()}

json1 = json.dumps(partition)
f1 = open(dataDirOut+"partition.json","w")
f1.write(json1)
f1.close()

json2 = json.dumps(partition_labels)
f2 = open(dataDirOut+"labels.json","w")
f2.write(json2)
f2.close()
