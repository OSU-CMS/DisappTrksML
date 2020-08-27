import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):

	def __init__(self, file_ids, batch_size=32, dim=(1000,4), n_channels=1, n_classes=10, shuffle=True):
		self.file_ids = file_ids
		self.batch_size = batch_size
		self.dim = dim
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		return int(np.floor(len(self.file_ids) / self.batch_size))

	def __getitem__(self, index):
		indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
		file_ids_temp = [self.file_ids[k] for k in indexes]
		X, y = self.__data_generation(file_ids_temp)
		return X, y

	def on_epoch_end(self):
		self.indexes = np.arrange(len(self.file_ids))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def __data_generation(self, file_indices):
		X = np.empty((self.batch_size, *self.dim, self.n_channels))
		y = np.empty((self.batch_size), dtype=int)

		# generate data
		for idx in file_indices:
			fin = np.load('E:\shared\images_SingleEle2017F\hist_' + idx + '.root.npz')
			X = np.concatenate((X, fin['images_reco']))
			X = np.concatenate((X, fin['images_fail']))
			y = np.concatenate((y, fin['labels_reco']))
			y = np.concatenate((y, fin['labels_fail']))

		p = np.random.permutation(len(X))
		X = X[p]
		y = y[p]

		return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
