import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import ndimage

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Finds the center of the character in the image
def getBestShift(img):
	cy, cx = ndimage.measurements.center_of_mass(img)

	rows, cols = img.shape
	shiftx = np.round(cols / 2.0 - cx).astype(int)
	shifty = np.round(rows / 2.0 - cy).astype(int)

	return shiftx, shifty


# Responsible to sift the image in the direction of sx, sy
def shift(img,sx,sy):
	rows,cols = img.shape
	M = np.float32([[1,0,sx],[0,1,sy]])
	shifted = cv2.warpAffine(img,M,(cols,rows))
	return shifted


# Given a png file, it will
def convertImage(file):
	# Did the followng to convert the image to an array.
	# Based on the EMNIST directions found online.
	x = cv2.imread(file, 0)
	x = np.invert(x)
	x = cv2.GaussianBlur(x, (5, 5), 10)
	x = cv2.resize(x, (28, 28), interpolation=cv2.INTER_CUBIC)

	# Now we rearange the image so it is at the center and add a padding
	while np.sum(x[0]) == 0:
		x = x[1:]

	while np.sum(x[:, 0]) == 0:
		x = np.delete(x, 0, 1)

	while np.sum(x[-1]) == 0:
		x = x[:-1]

	while np.sum(x[:, -1]) == 0:
		x = np.delete(x, -1, 1)
	rows, cols = x.shape

	if rows > cols:
		factor = 20.0 / rows
		rows = 20
		cols = int(round(cols * factor))
		x = cv2.resize(x, (cols, rows))
	else:
		factor = 20.0 / cols
		cols = 20
		rows = int(round(rows * factor))
		x = cv2.resize(x, (cols, rows))

	colsPadding = (int(np.math.ceil((28 - cols) / 2.0)), int(np.math.floor((28 - cols) / 2.0)))
	rowsPadding = (int(np.math.ceil((28 - rows) / 2.0)), int(np.math.floor((28 - rows) / 2.0)))
	x = np.lib.pad(x, (rowsPadding, colsPadding), 'constant')

	shiftx, shifty = getBestShift(x)
	shifted = shift(x, shiftx, shifty)
	x = shifted

	x = x.reshape(1, 28, 28, 1)
	x = x.astype('float32')
	x = x.flatten()
	x /= 255.0
	return x


# For testing purposes. Allows us to visualize the image through the array
def display_sample_from_train(num, images):
	image = images[num].reshape([28, 28])
	plt.title('Sample: %d' % num)
	plt.imshow(image, cmap=plt.get_cmap('gray_r'))
	plt.show()


def main():
	images = np.zeros((1, 784))  # images is of size 1 and stores only one image in the form of an np array of size 784

	images[0] = convertImage("my_drawing.png")  # my_drawing.png is the output of draw.py

	sess = tf.compat.v1.InteractiveSession()  # Starts the TensorFlow session

	# Import the model which was trained and saved in train.py
	new_saver = tf.compat.v1.train.import_meta_graph('my_test_model.meta')
	new_saver.restore(sess, tf.train.latest_checkpoint('./'))
	graph = tf.compat.v1.get_default_graph()

	# Import the variables and functions from train.py
	input_images = tf.compat.v1.placeholder(tf.float32, shape=[None, 784], name="input_images")
	target_labels = tf.compat.v1.placeholder(tf.float32, shape=[None, 47], name="target_labels")

	input_weights = graph.get_tensor_by_name("input_weights:0")
	input_biases = graph.get_tensor_by_name("input_biases:0")
	hidden_weights = graph.get_tensor_by_name("hidden_weights:0")
	hidden_biases = graph.get_tensor_by_name("hidden_biases:0")

	input_layer = tf.matmul(input_images, input_weights)
	hidden_layer = tf.nn.relu(input_layer + input_biases)
	digit_weights = tf.matmul(hidden_layer, hidden_weights) + hidden_biases

	sortedPrected = tf.argsort(digit_weights, 1, direction='DESCENDING')
	prediction = tf.argmax(digit_weights, 1)
	predictedArr = prediction.eval(feed_dict={input_images: images})  # Feed the image to the prediction to classify
	sortedPrectedArr = sortedPrected.eval(feed_dict={input_images: images})  # Get the predicted Array
	print(sortedPrectedArr)
	print(predictedArr[0])

	# Takes the integer value that prediction returns and converts it to the appropriate character
	final_arr = []
	for i in range(len(predictedArr)):
		label = predictedArr[i]
		if 9 < label < 36:
			label = chr(label + 55)
		elif 36 <= label <= 37:  # From a-b
			label = chr(label + 61)
		elif 38 <= label <= 42:  # From d-h
			label = chr(label + 62)
		elif label == 43:  # For n
			label = 'n'
		elif label == 44 or label == 45:  # From q to r
			label = chr(label - 44 + int('q'))
		elif label == 46:
			label = 't'
		else:
			label = chr(label + 48)
		final_arr.append(label)

	print(final_arr)

	return final_arr[0]


if __name__ == '__main__':
	main()
