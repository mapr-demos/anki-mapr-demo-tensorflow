import argparse
import os
import skimage.data
import skimage.transform
import tensorflow as tf
import numpy as np
import random
import sys

parser = argparse.ArgumentParser()

### Process Arguments ###
def f_addArguments():
	parser.add_argument(
		 '--train_dir'
		,type=str
		,default='/vagrant/Training'
		,help='Location of training images'
	)

	parser.add_argument(
		 '--image_ext'
		,type=str
		,default='jpg'
		,help='File ext of images'
	)

	parser.add_argument(
		 '--trains'
		,type=int
		,default='201'
		,help='Number of training iterations'
	)

	parser.add_argument(
		 '--resize_size'
		,type=int
		,default='32'
		,help='Pixel size to resize to'
	)

	parser.add_argument(
		 '--sample_size'
		,type=int
		,default='20'
		,help='Number of images to sample after training'
	)

	parser.add_argument(
		 '--learning_rate'
		,type=float
		,default='0.01'
		,help='Learning rate of optimizer'
	)

	parser.add_argument(
		 '--model_path'
		,type=str
		,default='/vagrant/trained_model'
		,help='Path to save models to'
	)

	parser.add_argument(
		 '--models'
		,type=int
		,default=1
		,help='Number of models to produce'
	)

	parser.add_argument(
		 '--min_loss'
		,type=float
		,default='0.1'
		,help='Minimum loss for a model to be considered as worth saving'
	)

### Load Data Files ###
def load_data_files(image_dir, image_ext):

	dirs = []
	for d in os.listdir(image_dir):
		if os.path.isdir(os.path.join(image_dir, d)):
			dirs.append(d)

	images = []
	labels = []
	for d in dirs:
#		print '>>> Dir: %s' % d
		label_dir = os.path.join(image_dir, d)
		files = [ os.path.join(label_dir, f)
			  for f in os.listdir(label_dir)
			  if f.endswith(image_ext) ]
#		print files
		for f in files:
			images.append(skimage.data.imread(f))
#			images.append(skimage.data.imread(f,True))
			labels.append(d)

	return images, labels, dirs

### Initialize Arguments ###
f_addArguments()
FLAGS, unparsed = parser.parse_known_args()
print vars(FLAGS)

### Load Training Data ###
### Keep the number of directories
images, labels, unqLabels = load_data_files(FLAGS.train_dir, FLAGS.image_ext)

print '>>> Lables: %d, Images: %d' % (len(unqLabels), len(images))
#for i in images:
#	print(">>> Shape: {0}, min: {1}, max: {2}".format(i.shape, i.min(), i.max()))

images_reszie = [ skimage.transform.resize(i, (FLAGS.resize_size, FLAGS.resize_size))
		for i in images ]
#for i in images_reszie:
#	print(">>> Shape: {0}, min: {1}, max: {2}".format(i.shape, i.min(), i.max()))

### Convert to an array ###
labels_a = np.array(labels)
images_a = np.array(images_reszie)
print("labels array shape: ", labels_a.shape, ", images array shape: ", images_a.shape)

# Start the TensorFlow processing
graph = tf.Graph()

with graph.as_default():

	# Placeholders for images
	images_ph = tf.placeholder(tf.float32, [None, FLAGS.resize_size, FLAGS.resize_size, 3])
	labels_ph = tf.placeholder(tf.int32, [None])	

	# Flatten the images
	images_flat = tf.contrib.layers.flatten(images_ph)

	# Create the fully connected layer
	logits = tf.contrib.layers.fully_connected(images_flat, len(unqLabels), tf.nn.relu)

	# Convert logits to label indexes (int)
	predicted_labels = tf.argmax(logits, 1)

	# Define loss function
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))

	# Create training op.
#	train = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)
	train = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)

	print(">>> images_flat: ", images_flat)
	print(">>> logits: ", logits)
	print(">>> loss: ", loss)
	print(">>> predicted_labels: ", predicted_labels)

	# Initialize function
	init = tf.global_variables_initializer()

	session = tf.Session()			# Get session

	for m in range(0, FLAGS.models):
		print (">>> Model: ", m)
		_ = session.run([init])			# Initialize

		# Train
		for i in range(FLAGS.trains):
			_, loss_value = session.run([train, loss], 
						    feed_dict={images_ph: images_a, labels_ph: labels_a})
			if i % 10 == 0:
				print("Step: ", i, " Loss: ", loss_value)

		# Using the model
		sample_indexes = random.sample(range(len(images_reszie)), FLAGS.sample_size)
		sample_images = [images_reszie[i] for i in sample_indexes]
		sample_labels = [labels[i] for i in sample_indexes]

		predicted = session.run([predicted_labels], feed_dict={images_ph: sample_images})[0]

		predicted_label_values = [unqLabels[l] for l in predicted ] 
		print(">>> ", sample_labels)
		print(">>> ",predicted_label_values)

		match_count = sum([int(y == y_) for y, y_ in zip(sample_labels, predicted_label_values)])
		accuracy = float(match_count) / float(FLAGS.sample_size)
		print (">>> {} from {} [ {:.3f} ]".format(match_count, FLAGS.sample_size, accuracy))

		# Save the model
		if loss_value < FLAGS.min_loss:
			print (">>> Saving model to ", FLAGS.model_path + "-" + str(loss_value))
			saver = tf.train.Saver()
			saver.save(session, FLAGS.model_path + "-" + str(loss_value))








