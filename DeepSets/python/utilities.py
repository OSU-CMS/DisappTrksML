import numpy as np
import matplotlib.pyplot as plt

img_max = 0.25
img_min = -0.25
img_pixels = 50

def displayImage(trackImage, detectorIndex=-1, colormap='cubehelix_r'):
	img = np.zeros(shape=(img_pixels, img_pixels))
	for hit in trackImage:
		if detectorIndex >= 0 and hit[3] != detectorIndex:
			continue
		ix = int((img_pixels - 1) / (img_max - img_min) * (hit[0] - img_min))
		iy = int((img_pixels - 1) / (img_max - img_min) * (hit[1] - img_min))
		if ix < 0 or ix >= img_pixels or iy < 0 or iy >= img_pixels:
			continue
		img[ix, iy] += hit[2]
	plt.imshow(img, cmap=colormap)

def displayTrainingHistory(history):
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']

	loss = history.history['loss']
	val_loss = history.history['validation_loss']

	epochs = range(1, len(acc) + 1)

	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()

	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()

	plt.show()
