import argparse
import os
import skimage.data
import skimage.transform
import tensorflow as tf
import numpy as np
import random
import sys
import time
import requests

parser = argparse.ArgumentParser()

### Process Arguments ###
def f_addArguments():
	parser.add_argument(
		 '--test_dir'
		,type=str
		,default='/vagrant/Testing'
		,help='Location of test images'
	)

	parser.add_argument(
		 '--image_ext'
		,type=str
		,default='jpg'
		,help='File ext of images'
	)

	parser.add_argument(
		 '--poll_frq'
		,type=int
		,default='5'
		,help='Polling frequency (seconds)'
	)

	parser.add_argument(
		 '--resize_size'
		,type=int
		,default='32'
		,help='Amount to resize to'
	)

	parser.add_argument(
		 '--model_path'
		,type=str
		,default='/vagrant/trained_model'
		,help='Path to saved model'
	)

	parser.add_argument(
		 '--anki_address'
		,type=str
		,default='http://localhost:7877'
		,help='Address of anki controller'
	)

	parser.add_argument(
		 '--use_camera'
		,type=int
		,default='0'
		,help='Use camera (0 for no, 1 for yes)'
	)

	parser.add_argument(
		 '--use_anki'
		,type=int
		,default='0'
		,help='Use anki (0 for no, 1 for yes)'
	)

### Load Data Files ###
def load_data_files(image_dir, image_ext):

	dirs = []
	for d in os.listdir(image_dir):
		if os.path.isdir(os.path.join(image_dir, d)):
			dirs.append(d)

	images = []
	labels = []
	allFiles = []
	for d in dirs:
#		print '>>> Dir: %s' % d
		label_dir = os.path.join(image_dir, d)
		files = [ os.path.join(label_dir, f)
			  for f in os.listdir(label_dir)
			  if f.endswith(image_ext) ]
#		print files
		for f in files:
			images.append(skimage.data.imread(f))
			labels.append(d)
			allFiles.append(f)

	return images, labels, dirs, allFiles

### Initialize Arguments ###
f_addArguments()
FLAGS, unparsed = parser.parse_known_args()
print vars(FLAGS)

urlScan = FLAGS.anki_address + "/rescan"
urlConnect = FLAGS.anki_address + "/startDemoConnect"
urlGo = FLAGS.anki_address + "/startDemoGo"
urlGoFast = FLAGS.anki_address + "/demoGoFast"
urlStop = FLAGS.anki_address + "/demoStop"
print urlScan, urlConnect, urlGo, urlGoFast, urlStop

label_text = [ 'Stop', 'Go', 'Connect', 'Scan', 'Fast', 'None' ]

# Call this one so that unqLabels is populated
images, labels, unqLabels, allFiles = load_data_files(FLAGS.test_dir, FLAGS.image_ext)

with tf.Session() as session:

	# Placeholders for images
	images_ph = tf.placeholder(tf.float32, [None, FLAGS.resize_size, FLAGS.resize_size, 3])
#	labels_ph = tf.placeholder(tf.int32, [None])	

	# Flatten the images
	images_flat = tf.contrib.layers.flatten(images_ph)

	# Create the fully connected layer
	logits = tf.contrib.layers.fully_connected(images_flat, len(unqLabels), tf.nn.relu)

	# Convert logits to label indexes (int)
	predicted_labels = tf.argmax(logits, 1)

	# Load the model
	print (">>> Loading model ", FLAGS.model_path)
	saver = tf.train.Saver()
	saver.restore(session, FLAGS.model_path)

	while True:
		# Load Data
		# Note the directory should all the required labels
		if (FLAGS.use_camera == 1):
#			print (time.strftime("%H:%M:%S: ") + ">>> Taking picture ...")
			os.system("fswebcam --no-banner --quiet " + FLAGS.test_dir + "/0/capture-" + time.strftime("%H-%M-%S") + "." + FLAGS.image_ext + " 2>/tmp/webcamerrs.out")

#		print (time.strftime("%H:%M:%S: ") + ">>> Checking for images...")
		images, labels, unqLabels, allFiles = load_data_files(FLAGS.test_dir, FLAGS.image_ext)
		if len(images) > 0:

			for f in range(0,len(allFiles)):
				images_reszie = [ skimage.transform.resize(images[f], (FLAGS.resize_size, FLAGS.resize_size)) ]
				images_a = np.array(images_reszie)
				predicted = session.run([predicted_labels], feed_dict={images_ph: images_a})[0]
				print (time.strftime("%H:%M:%S: ") + '>>> Image: ' + allFiles[f] + ' Predicted: ' + label_text[predicted[0]] + ' (' + str(predicted[0]) + ')')
				os.remove(allFiles[f])
				try:
					if label_text[predicted[0]] == "Scan":
						if (FLAGS.use_anki == 1):
							url = urlScan
#							print(">>> Calling anki ..." + url)
							response = requests.post(url, timeout=0.5)
					if label_text[predicted[0]] == "Connect":
						if (FLAGS.use_anki == 1):
							url = urlConnect
#							print(">>> Calling anki ..." + url)
							response = requests.get(url, timeout=0.5)
					if label_text[predicted[0]] == "Go":
						if (FLAGS.use_anki == 1):
							url = urlGo
#							print(">>> Calling anki ..." + url)
							response = requests.get(url, timeout=0.5)
					if label_text[predicted[0]] == "Fast":
						if (FLAGS.use_anki == 1):
							url = urlGoFast
#							print(">>> Calling anki ..." + url)
							response = requests.get(url, timeout=0.5)
					if label_text[predicted[0]] == "Stop":
						if (FLAGS.use_anki == 1):
							url = urlStop
#							print(">>> Calling anki ..." + url)
							response = requests.get(url, timeout=0.5)
				except requests.exceptions.RequestException as e:
					print e

#		print (">>> Sleeping for: " + str(FLAGS.poll_frq) + " seconds")
		time.sleep(FLAGS.poll_frq)






