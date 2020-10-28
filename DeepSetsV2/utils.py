import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial
from itertools import repeat
import numpy as np
import os, sys
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import pickle

class bcolors:
		HEADER = '\033[95m'
		BLUE = '\033[94m'
		GREEN = '\033[92m'
		YELLOW = '\033[93m'
		RED = '\033[91m'
		ENDC = '\033[0m'
		BOLD = '\033[1m'
		UNDERLINE = '\033[4m'

def nested_defaultdict(default_factory, depth=1):
	result = partial(defaultdict, default_factory)
	for _ in repeat(None, depth - 1):
			result = partial(defaultdict, result)
	return result()

def save_event(x,outf="event.png"):

		if(x.shape[0] == 404): x = x[4:]
		if(x.shape[0] == 400): x = np.reshape(x, (100,4))

		fig, axs = plt.subplots(1,3, figsize=(17,5))
				
		for i in range(3):
			ax = axs[i]
			im = x[np.where(x[:,3]==i)]
			im = im[np.where(im[:,2]!=0)]
			h = ax.hist2d(im[:,0],im[:,1],weights=im[:,2],range=[[-0.3,0.3],[-0.3,0.3]],cmap='cubehelix',bins=(80,80))
			fig.colorbar(h[3], ax = ax)
			ax.set_xlabel("Eta")
			ax.set_ylabel("Phi")
			i+=1

		axs[0].set_title("ECAL")
		axs[1].set_title("HCAL")
		axs[2].set_title("MUO")

		plt.tight_layout()
		plt.savefig(outf)
		plt.cla()
		plt.close(fig)


def plot_history(history, plotDir, variables=['accuracy','loss']):
	for var in variables:
		plt.plot(history.history[var],label='train')
		plt.plot(history.history['val_'+var],label='test')
		plt.title(var + ' History')
		plt.ylabel(var)
		plt.xlabel('Epoch')
		plt.legend()
		plt.savefig(plotDir+var+'_history.png')
		plt.clf()
		
def calc_cm_one_hot(y_test,predictions):
		confusion_matrix = nested_defaultdict(int,2)
		class_true, class_predictions = [], []
		for t in y_test:
			for i,elm in enumerate(t): 
				if(elm == 1): 
					class_true.append(i)
		for p in predictions:
			for i,elm in enumerate(p): 
				if(elm == 1): 
					class_predictions.append(i)
		for t,p in zip(class_true, class_predictions):
				confusion_matrix[t][p] += 1
		return confusion_matrix

def calc_cm(true,predictions):
		confusion_matrix = nested_defaultdict(int,2)
		for t,p in zip(true, predictions):
				confusion_matrix[t][p] += 1
		return confusion_matrix

def plot_certainty_one_hot(y_test,predictions,f):

		correct_certainty, notcorrect_certainty = [],[]
		for true,pred in zip(y_test, predictions):
				if np.argmax(true) == np.argmax(pred):
						correct_certainty.append(pred[np.argmax(pred)])
				else:
						notcorrect_certainty.append(pred[np.argmax(pred)])
		
		plt.hist(correct_certainty,alpha=0.5,label='Predicted Successfully',density=True)
		plt.hist(notcorrect_certainty,alpha=0.5,label='Predicted Unsuccessfully',density=True)
		plt.title("Certainty")
		plt.legend()
		plt.savefig(f)
		plt.clf()

def plot_confusion_matrix(confusion_matrix, target_names, f='cm.png', title='Confusion Matrix', cmap="Blues"):
		
		#convert to array of floats
		cm = np.zeros([2,2])
		for i in range(2):
				for j in range(2):
						cm[i][j] = confusion_matrix[i][j]
		cm = cm.astype(float)

		plt.imshow(cm,cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(target_names))
		plt.xticks(tick_marks, target_names)
		plt.yticks(tick_marks, target_names)
		plt.axis('equal')
		plt.tight_layout()

		width, height = cm.shape
		for x in range(width):
				for y in range(height):
						plt.annotate(str(cm[x][y]), xy=(y, x), 
												horizontalalignment='center',
												verticalalignment='center')
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.savefig(f, bbox_inches = "tight")
		plt.clf()

def calc_binary_metrics(confusion_matrix):
		c1=1
		c2=0
		TP = confusion_matrix[c1][c1]
		FP = confusion_matrix[c2][c1]
		FN = confusion_matrix[c1][c2]
		TN = confusion_matrix[c2][c2]

		if((TP+FP) == 0): precision = 0
		else: precision = TP / (TP + FP)
		if((TP+FN) == 0): recall = 0
		else: recall = TP / (TP + FN)

		return precision, recall

def plot_grid(gs, x_label, y_label, x_target_names, y_target_names, title = 'Grid Search', f='gs.png', cmap=plt.get_cmap('Blues')):
 
		#convert to array of floats
		grid = np.zeros([len(y_target_names),len(x_target_names)])
		for i in range(len(y_target_names)):
				for j in range(len(x_target_names)):
						grid[i][j] = round(gs[i][j],3)
		grid = grid.astype(float)

		plt.imshow(grid, interpolation='nearest',cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks_x = np.arange(len(x_target_names))
		tick_marks_y = np.arange(len(y_target_names))
		plt.xticks(tick_marks_x, x_target_names)
		plt.yticks(tick_marks_y, y_target_names)
		
		plt.tight_layout()

		width, height = grid.shape

		for x in range(width):
				for y in range(height):
						plt.annotate(str(grid[x][y]), xy=(y, x), 
												horizontalalignment='center',
												verticalalignment='center')

		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.savefig(f, bbox_inches='tight')
		plt.clf()

# plot precision and recall for different electron probabilities
def plot_precision_recall(true, probas, fname, nsplits=20):
	precisions, recalls, splits = [],[],[]
	for split in np.arange(0,1,1.0/nsplits):
		preds = []
		for p in probas:
				if(p>split): preds.append(1)
				else: preds.append(0)
		cm = calc_cm(true,preds)
		precision, recall = calc_binary_metrics(cm)
		precisions.append(precision)
		recalls.append(recall)
		splits.append(split)

	plt.scatter(splits, precisions, label="Precision", marker="+",color="red")
	plt.scatter(splits, recalls, label="Recall", marker="+",color="blue")
	plt.ylim(-0.1,1.1)
	plt.xlim(-0.1,1.1)
	plt.ylabel("Metric Value")
	plt.xlabel("Split on Electron Probability")
	plt.title("Metrics per Probability")
	plt.legend()
	plt.tight_layout()
	plt.savefig(fname)
	plt.clf()

# calculate and save metrics
def metrics(true, predictions, plotDir, threshold=0.5):

	class_predictions = []
	for p in predictions:
			if(p >= threshold): 
				class_predictions.append(1)
			else: 
				class_predictions.append(0)

	cm = calc_cm(true, class_predictions)

	print(cm)

	plot_confusion_matrix(cm,['bkg','e'],plotDir + 'cm.png')
	
	plot_precision_recall(true,predictions,plotDir+'precision_recall.png',nsplits=50)

	precision, recall = calc_binary_metrics(cm)
	auc = roc_auc_score(true, predictions)

	fpr, tpr, thresholds = roc_curve(true, predictions)
	plt.plot(fpr,tpr)
	plt.title("ROC")
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.savefig(plotDir+"roc.png")
	plt.clf()
	np.savez_compressed(plotDir+"roc.npz",fpr=fpr,tpr=tpr)

	fileOut = open(plotDir+"metrics.txt","w")
	fileOut.write("Precision = TP/(TP+FP) = fraction of predicted true actually true "+str(round(precision,5))+"\n")
	fileOut.write("Recall = TP/(TP+FN) = fraction of true class predicted to be true "+str(round(recall,5))+"\n")
	fileOut.write("AUC Score:"+str(round(auc,5)))
	fileOut.close()


def chunks(lst, chunk_sizes):
	out = []
	i = 0
	for chunk in chunk_sizes:
			out.append(lst[i:i+chunk])
			i=i+chunk
	return out

def make_batches(events, files, nPerBatch, nBatches):

		event_batches_full = chunks(events,nPerBatch)
		file_batches_full = chunks(files,nPerBatch)

		event_batches, file_batches = [], []
		batches = 0

		for events, files in zip(event_batches_full, file_batches_full):
				if(batches == nBatches): break
				if(len(events)==0):
					event_batches.append([-1])
					file_batches.append([-1])
				else:
					event_batches.append([events[0],events[-1]])
					temp = []
					for file in files:
						if(file not in temp): temp.append(file)
					file_batches.append(temp)
				batches+=1

		return np.array(event_batches), file_batches

def count_events(file_batches, event_batches, dict):

	nSaved=0
	for files, indices in zip(file_batches, event_batches):
			if(len(files) == 1 and files[0] == -1): continue
			lastFile = len(files)-1
			for iFile, file in enumerate(files):
					if(iFile == 0 and iFile != lastFile):
							nSaved+=(dict[file]-indices[0])

					elif(iFile == lastFile and iFile != 0):
							nSaved+=(indices[1]+1)

					elif(iFile == 0 and iFile == lastFile):
							nSaved+=(indices[1]-indices[0]+1)

					elif(iFile != 0 and iFile != lastFile):
							nSaved+=dict[file]
	return nSaved

def prepare_data(dataDir, nTotE, batch_size=64, val_size=0.2, undersample_bkg=-1):

	# import count dicts
	with open(dataDir+'sCounts.pkl', 'rb') as f:
		sCounts = pickle.load(f)
	with open(dataDir+'bkgCounts.pkl', 'rb') as f:
		bkgCounts = pickle.load(f)

	# count how many events are in the files for each class
	availableE = sum(list(sCounts.values()))
	availableBkg = sum(list(bkgCounts.values()))

	# fractions for each class for the total dataset
	fE = availableE*1.0/(availableE + availableBkg)
	fBkg = availableBkg*1.0/(availableE + availableBkg)

	# calculate how many total background events for the requested electrons
	# to keep the same fraction of events, or under sample
	nTotBkg = int(nTotE*1.0*availableBkg/availableE)
	if(undersample_bkg!=-1): nTotBkg = int(nTotE*1.0*undersample_bkg/(1-undersample_bkg))

	# can't request more events than we have
	if(nTotE > availableE): sys.exit("ERROR: Requested more signal events than are available")
	if(nTotBkg > availableBkg): sys.exit("ERROR: Requested more background events than available")

	# batches per epoch
	nBatches = int(np.floor((nTotE + nTotBkg)*1.0/batch_size))

	# count how many e/bkg events in each batch
	ePerBatch, bkgPerBatch = np.zeros(nBatches), np.zeros(nBatches)
	iBatch = 0
	while np.sum(ePerBatch) < nTotE:
		ePerBatch[iBatch]+=1
		iBatch+=1
		if(iBatch == nBatches): iBatch = 0
	for iBatch in range(nBatches):
		toAdd = int(batch_size - ePerBatch[iBatch])
		if(toAdd == 0): continue
		for j in range(toAdd):
			bkgPerBatch[iBatch]+=1
			if(np.sum(bkgPerBatch) == nTotBkg): 
				print("Error: not enough background events.")
				sys.exit(0)
		if(np.sum(bkgPerBatch) == nTotBkg): break

	for iBatch in range(nBatches):
		if(ePerBatch[iBatch]+bkgPerBatch[iBatch]!=batch_size): 
			print("Error: batches aren't filled properly.")
			sys.exit(0)
	ePerBatch = ePerBatch.astype(int)
	bkgPerBatch = bkgPerBatch.astype(int)

	# fill lists of all events and files
	b_events, b_files = [], []
	for file, nEvents in bkgCounts.items():
		for evt in range(nEvents):
			b_events.append(evt)
			b_files.append(file)
	e_events, e_files = [], []
	for file, nEvents in sCounts.items():
		for evt in range(nEvents):
			e_events.append(evt)
			e_files.append(file)

	# make batches
	bkg_event_batches, bkg_file_batches = make_batches(b_events, b_files, bkgPerBatch, nBatches)
	e_event_batches, e_file_batches = make_batches(e_events, e_files, ePerBatch, nBatches)
	
	# train/validation split
	train_e_event_batches, val_e_event_batches, train_e_file_batches, val_e_file_batches = train_test_split(e_event_batches, e_file_batches, test_size=val_size, random_state=42)
	train_bkg_event_batches, val_bkg_event_batches, train_bkg_file_batches, val_bkg_file_batches = train_test_split(bkg_event_batches, bkg_file_batches, test_size=val_size, random_state=42)

	# count events in each batch
	nSavedETrain = count_events(train_e_file_batches, train_e_event_batches, sCounts)
	nSavedEVal = count_events(val_e_file_batches, val_e_event_batches, sCounts)
	nSavedBkgTrain = count_events(train_bkg_file_batches, train_bkg_event_batches, bkgCounts)
	nSavedBkgVal = count_events(val_bkg_file_batches, val_bkg_event_batches, bkgCounts)

	# add background events to validation data
	# to keep ratio e/bkg equal to that in original dataset
	if(abs(1-nSavedEVal*1.0/(nSavedEVal+nSavedBkgVal)/fE) > 0.05):
		nBkgToLoad = int(nSavedEVal*(1-fE)/fE-nSavedBkgVal)
		lastFile = bkg_file_batches[-1][-1]

		b_events, b_files = [], []
		reached = False
		for file, nEvents in bkgCounts.items():
			if(int(file) != lastFile and not reached): continue
			else: reached = True

			for evt in range(nEvents):
				b_events.append(evt)
				b_files.append(file)

		# make batches of same size with bkg files
		nBatchesAdded = int(nBkgToLoad*1.0/batch_size)
		bkgPerBatch = [batch_size]*nBatchesAdded
				 
		bkg_event_batches_added, bkg_file_batches_added = make_batches(b_events, b_files, bkgPerBatch, nBatchesAdded)

		nAddedBkg = count_events(bkg_file_batches, bkg_event_batches, bkgCounts)

		# add the bkg and e events to rebalance val data
		filler_events = [[0,0]]*nBatchesAdded
		filler_files = [list(set([-1])) for _ in range(nBatchesAdded)]

		val_bkg_event_batches = np.concatenate((val_bkg_event_batches,bkg_event_batches_added))
		val_bkg_file_batches = val_bkg_file_batches + bkg_file_batches_added
		val_e_event_batches = np.concatenate((val_e_event_batches,filler_events))
		val_e_file_batches = val_e_file_batches + filler_files

		# re count
		nSavedEVal = count_events(val_e_file_batches, val_e_event_batches, sCounts)
		nSavedBkgVal = count_events(val_bkg_file_batches, val_bkg_event_batches, bkgCounts)

	print("\t\tElectrons\tBackground\te/(e+bkg)")
	print("Requested:\t"+str(nTotE)+"\t\t"+str(nTotBkg)+"\t\t"+str(round(nTotE*1.0/(nTotE+nTotBkg),5)))
	print("Training on:\t"+str(nSavedETrain)+"\t\t"+str(nSavedBkgTrain)+"\t\t"+str(round(nSavedETrain*1.0/(nSavedETrain+nSavedBkgTrain),5)))
	print("Validating on:\t"+str(nSavedEVal)+"\t\t"+str(nSavedBkgVal)+"\t\t"+str(round(nSavedEVal*1.0/(nSavedEVal+nSavedBkgVal),5)))
	print("Dataset:\t"+str(availableE)+"\t\t"+str(availableBkg)+"\t\t"+str(round(fE,5)))

	return [train_e_file_batches, train_e_event_batches, val_e_file_batches, val_e_event_batches], [train_bkg_file_batches, train_bkg_event_batches,val_bkg_file_batches, val_bkg_event_batches]