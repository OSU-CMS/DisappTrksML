import numpy as np 

files = ['output_higgsino_700_10.npz',
		'output_higgsino_700_100.npz',
		'output_higgsino_700_1000.npz',
		'output_higgsino_700_10000.npz',
		'output_electrons.npz']
for file in files:
	print file
	infile = np.load(file)
	results = infile['results']
	print results[results[:,0] > -1].shape