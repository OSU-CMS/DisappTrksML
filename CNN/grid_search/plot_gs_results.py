import numpy as np
import matplotlib.pyplot as plt
from itertools import product 
from utils import *

plotDir = "/data/users/llavezzo/cnn/plots/"

layers_list = [2,5,10]
filters_list = [32, 64, 128]  
optimizers = ['adam','adadelta']
parameters = list(product(layers_list, filters_list, optimizers))
metrics = ['Accuracy','Loss', 'Iterations', 'Precision', 'Recall', 'AUC']

nMetrics = 6
results_array = []
for i in range(len(parameters)):
	temp = np.load(plotDir+'gsresults_'+str(i)'.npy')
	results_array.append(temp)

for iMetric,metric in enumerate(metrics):
	for optimizer in optimizers:
		grid = np.zeros([len(layers_list),len(filters_list)])
		for i,row in enumerate(results_array):
			if(parameters[i][2] == optimizer):

				iFilter = filters_list.index(parameters[i][1])
				iLayer = layers_list.index(parameters[i][0])
				grid[iLayer,iFilter] = row[iMetric]

		plot_grid(grid, "Filters", "Layers",
        filters_list, layers_list, metric + ' Grid Search with ' + optimizer, 
        plotDir + metric + '_' + optimizer + '.png')