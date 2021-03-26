import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from DisappTrksML.SiameseNetwork.MuonModel import *

# build model from architecture
arch = MuonModel(eta_range=1.0,phi_range=1.0,max_hits=20)
arch.buildModel()

# load in the data, combine the classes in a multidimensional numpy array with nEvents each
data = np.load("test.npy", allow_pickle=True)
nEvents = 100
X = None
for i in range(4):
	iClass = data[i][:nEvents]
	iClass = np.reshape(iClass, (1, nEvents, 20, 7))
	if X is None: X = iClass
	else: X = np.vstack((X, iClass)) 
data = np.load("val_data.npy", allow_pickle=True)
nEvents = 30
X_val = None
for i in range(2):
	iClass = data[i][:nEvents]
	iClass = np.reshape(iClass, (1, nEvents, 20, 7))
	if X_val is None: X_val = iClass
	else: X_val = np.vstack((X_val, iClass)) 
print(X.shape)
print(X_val.shape)
arch.add_data(X, X_val)

# parameters
evaluate_every = 1000 # interval for evaluating on one-shot tasks
batch_size = 2
n_iter = 10000 # No. of training iterations
N_way = 3 # how many classes for testing one-shot tasks
N_way_val = 2
n_val = 15 # how many one-shot tasks to validate on
best = -1

train_accs, val_accs = [], []

# training
for i in range(1, n_iter+1):
    # print("i=",i)
	
	# Get a new batch
    (inputs,targets) = arch.get_batch(batch_size)
    loss = arch.model.train_on_batch(inputs, targets)
	
	# Check the training and validation performance
    if i % evaluate_every == 0:

    	print "\n ------------- \n"
        print "i: {0}".format(i)
        print "Train Loss: {:.4f}".format(loss)

		# Now get N-way test results for train and validation sets
        n_correct_train = 0
        n_correct_val = 0
        for testTrials in range(n_val):

			# train performance
            inputs, targets = arch.make_oneshot(N_way, use_test_data=False)
            probs = arch.model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct_train += 1

			# validation performace
            inputs, targets = arch.make_oneshot(N_way_val, use_test_data=True)
            probs = arch.model.predict(inputs)
            print(probs, targets)
            if np.argmax(probs) == np.argmax(targets):
                n_correct_val += 1
        
        train_acc = (100.0 * n_correct_train / n_val)
        train_accs.append(train_acc)
        print "Train Accuracy: {:.2f}".format(train_acc)

        val_acc = (100.0 * n_correct_val / n_val)
        val_accs.append(val_acc)
        print "Validation Accuracy: {:.2f}".format(val_acc)

        if val_acc >= best:
            print "Current best: {:.2f}, previous best: {:.2f}".format(val_acc, best)
            best = val_acc
            # model.save('models/new_siam_model.{}.h5'.format(i))

plt.plot(train_accs, label="train acc")
plt.plot(val_accs, label="validation acc")
plt.legend()
plt.savefig("history.png")