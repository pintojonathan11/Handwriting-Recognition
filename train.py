import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.io import loadmat

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def rotate(img):
	flipped = np.fliplr(img.reshape([28, 28]))
	return np.rot90(flipped)


def display_sample_from_train(num):
	print(y_train[num])
	label = y_train[num].argmax(axis=0)

	image = x_train[num].reshape([28, 28])
	plt.title('Sample: %d Label: %d' % (num, label))
	plt.imshow(image, cmap=plt.get_cmap('gray_r'))
	plt.show()


def display_sample_from_test(num):
	print(y_test[num].argmax(axis=0))
	label = y_test[num].argmax(axis=0)

	if 9 < label < 36:
		label = chr(label + 55)
	elif label >= 36:
		label = chr(label + 62)
	else:
		label = chr(label + 48)

	image = x_test[num].reshape([28, 28])
	plt.title('Sample: ' + str(num) + " Label: " + str(label))
	plt.imshow(image, cmap=plt.get_cmap('gray_r'))
	plt.show()


sess = tf.compat.v1.InteractiveSession()



# Load convoluted list structure form loadmat
mat_file_path = "./EMNIST/emnist-bymerge.mat"
mat = loadmat(mat_file_path)

# Load char mapping
mapping = {kv[0]: kv[1:][0] for kv in mat['dataset'][0][0][2]}

# Load training data
max_ = len(mat['dataset'][0][0][0][0][0][0])
x_train = mat['dataset'][0][0][0][0][0][0][:max_]
train_images = x_train.reshape(max_, 784)
y_train = mat['dataset'][0][0][0][0][0][1][:max_]

print("Training Size: " + str(max_))

# Load testing data
max_ = len(mat['dataset'][0][0][1][0][0][0])
x_test = mat['dataset'][0][0][1][0][0][0][:max_]
test_images = x_test.reshape(max_, 784)
y_test = mat['dataset'][0][0][1][0][0][1][:max_]

print("Testing Size: " + str(max_))

# Reshape training data to be valid
_len = len(train_images)
for i in range(len(train_images)):
	train_images[i] = rotate(train_images[i]).reshape(784)

# Reshape testing data to be valid
_len = len(test_images)
for i in range(len(test_images)):
	test_images[i] = rotate(test_images[i]).reshape(784)

train_images = train_images.astype("float32")
test_images = test_images.astype("float32")

x_train, x_test = train_images / 255.0, test_images / 255.0

cat_size = len(mapping)

y_train = tf.keras.utils.to_categorical(y_train, cat_size)  # 47 categories
y_test = tf.keras.utils.to_categorical(y_test, cat_size)

input_images = tf.compat.v1.placeholder(tf.float32, shape=[None, 784], name="input_images")
target_labels = tf.compat.v1.placeholder(tf.float32, shape=[None, cat_size], name="target_labels")

hidden_nodes = 10000  # 10000 recommended by EMNIST research paper

input_weights = tf.Variable(tf.random.truncated_normal([784, hidden_nodes]), name="input_weights")
input_biases = tf.Variable(tf.zeros([hidden_nodes]), name="input_biases")

hidden_weights = tf.Variable(tf.random.truncated_normal([hidden_nodes, cat_size]), name="hidden_weights")
hidden_biases = tf.Variable(tf.zeros([cat_size]), name="hidden_biases")

input_layer = tf.matmul(input_images, input_weights)
hidden_layer = tf.nn.relu(input_layer + input_biases)
digit_weights = tf.matmul(hidden_layer, hidden_weights) + hidden_biases

# cross entropy since it is harsher on wrong decisions
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=digit_weights, labels=target_labels))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss_function)  # 0.1 is the learning rate

# Checks if predictions are equal
prediction = tf.argmax(digit_weights, 1)
correct_prediction = tf.equal(tf.argmax(digit_weights, 1), tf.argmax(target_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Now we run it
tf.compat.v1.global_variables_initializer().run()

EPOCH = 10
BATCH_SIZE = 256
TRAIN_DATASIZE,_ = x_train.shape[0], x_train.shape[1]
print(TRAIN_DATASIZE)
PERIOD = TRAIN_DATASIZE // BATCH_SIZE

for e in range(EPOCH):
	idxs = np.random.permutation(TRAIN_DATASIZE)
	X_random = x_train[idxs]
	Y_random = y_train[idxs]

	for i in range(PERIOD):
		batch_X = X_random[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
		batch_Y = Y_random[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

		optimizer.run(feed_dict={input_images: batch_X, target_labels: batch_Y})

	print("Training epoch: " + str(e + 1))
	print("Accuracy: " + str(accuracy.eval(feed_dict={input_images: x_test, target_labels: y_test})))

predictedArr = prediction.eval(feed_dict={input_images: x_test[:10]})
print(predictedArr)

saver = tf.compat.v1.train.Saver([input_weights, input_biases, hidden_weights, hidden_biases])
saver.save(sess, 'my_test_model')