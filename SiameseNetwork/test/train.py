import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys

from DisappTrksML.SiameseNetwork.SiameseNetwork import *

# build model from architecture
arch = SiameseNetwork(
				phi_layers = [64,32], f_layers=[32],
				learning_rate = 0.1,
				eta_range=1.0,phi_range=1.0,max_hits=20)
arch.buildModel()

# load in the data, combine the classes in a multidimensional numpy array with nEvents each
X_train = arch.load_data("train_data.npy", nEvents = 100, nClasses=4)
X_val = arch.load_data("val_data.npy", nEvents = 25, nClasses=2)
X_ref = arch.load_data("ref_data.npy", nEvents = 25, nClasses=2)
arch.add_data(X_train, X_val, X_ref)

# parameters
evaluate_every = 10000 # interval for evaluating on one-shot tasks
batch_size = 2
n_iter = 100000 # No. of training iterations
N_way = 4 # how many classes for testing one-shot tasks
N_way_val = 2
n_val = 15 # how many one-shot tasks to validate on
best = -1

train_accs, val_accs = [], []
n_perClass, n_correct_perClass = np.zeros(2), np.zeros(2)

# training
for i in range(1, n_iter+1):
	
	# Get a new batch and train on batch
	(inputs,targets) = arch.get_batch(batch_size)
	loss = arch.model.train_on_batch(inputs, targets)
	
	# Check the training and validation performance
	if i % evaluate_every == 0:

		print "\n ------------- \n"
		print "i: {0}".format(i)
		print "Train Loss: {:.4f}".format(loss)

		# get N-way test results for train and validation sets
		n_correct_train = 0
		n_correct_val = 0
		for testTrials in range(n_val):

			# train performance
			inputs, targets, test_class = arch.make_oneshot(N_way, use_test_data=False, use_ref_set=False)
			probs = arch.model.predict(inputs)
			if np.argmax(probs) == np.argmax(targets):
				n_correct_train += 1

			# validation performace
			inputs, targets, test_class = arch.make_oneshot(N_way_val, use_test_data=True, use_ref_set=True)
			probs = arch.model.predict(inputs)
			n_perClass[test_class] += 1
			if np.argmax(probs) == np.argmax(targets):
				n_correct_val += 1
				n_correct_perClass[test_class] += 1

		n_correct = [0,0]
		N = X_val.shape[1]
		for iClass, events in enumerate(X_val):

			for event in events:

				# test one event at a time against the reference set
				test_images = np.asarray([event]*N)

				# similarity score with other class
				other_class_preds = arch.model.predict([test_images, X_ref[(iClass + 1)%2, :, :, :]])

				# similiarity score with same class
				same_class_preds = arch.model.predict([test_images,  X_ref[iClass, :, :, :]])

				# higher similiarity score wins
				if np.mean(same_class_preds) > np.mean(other_class_preds):
					n_correct[iClass] +=1 
	
		print(n_correct)
		if 0 in n_correct:
			print(same_class_preds)
			print(other_class_preds)
			print(probs)
			print(targets)

		train_acc = (100.0 * n_correct_train / n_val)
		train_accs.append(train_acc)
		print "Train Accuracy: {:.2f}".format(train_acc)

		val_acc = (100.0 * n_correct_val / n_val)
		val_accs.append(val_acc)
		print "Validation Accuracy: {:.2f}".format(val_acc)

		if val_acc >= best:
			print "Current best: {:.2f}, previous best: {:.2f}".format(val_acc, best)
			best = val_acc
			arch.model.save('models/siam_model.{}.h5'.format(i))

plt.plot(train_accs, label="train acc")
plt.plot(val_accs, label="validation acc")
plt.legend()
plt.savefig("history.png")

print "\n"
for i in range(2):
	print "Class {0} Accuracy: {1}".format(i, round(n_correct_perClass[i]/n_perClass[i],4))
