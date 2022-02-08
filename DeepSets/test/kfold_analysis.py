import os
import numpy as np
import pickle 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# loop through folders (expects metrics.pkl in each) and plot val_acc and val_loss
def get_kfold_metrics(folders):

	indices, val_losses, val_accs = [], [], []

	for folder in folders:

		fname = folder+'/kfold_metrics.pkl'
		if not os.path.isfile(fname):
			print(("Could not find",fname,". Skipping to next folder."))
			continue
		with open(fname,'rb') as f:
			metrics = pickle.load(f)
		indices.append(folder)
		val_losses.append(metrics['val_loss'])
		val_accs.append(metrics['val_accuracy'])

	return indices, [val_losses, val_accs]

def plot_kfold_metrics(indices, metrics, label="val_loss"):

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)

	ax.scatter(np.arange(len(metrics)), metrics)
	ax.set_title("kfold "+label)
	ax.set_xlabel("Parameter Index")
	ax.set_xticks(np.arange(len(indices)))
	ax.set_xticklabels(indices)
	plt.setp(ax.get_xticklabels(), rotation=90)
	fig.savefig("kfold_results_"+str(label)+".png", bbox_inches='tight')

def plot_history(infile,outfile):

	with open(infile,'rb') as f:
		history = pickle.load(f)

	loss = history['loss']
	val_loss = history['val_loss']

	epochs = list(range(1, len(loss) + 1))

	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()

	plt.savefig(outfile)


#####################

indices, metrics = get_kfold_metrics(["kfold"+str(i) for i in range(0,10)] + ["kfold"+str(i)+"_noBatchNorm" for i in range(10,25)])
plot_kfold_metrics(indices, metrics[0], "loss")
plot_kfold_metrics(indices, metrics[1], "acc")

min_i = np.array(indices)[np.argsort(metrics[0])]
min_l = np.array(metrics[0])[np.argsort(metrics[0])]

length = 1.5*max([len(i) for i in min_i])
for i in range(len(min_i)):
	if i > 5: break
	print(str(min_i[i]) + " "*int((length-len(min_i[i]))) + str(round(min_l[i],4)))