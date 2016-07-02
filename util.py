# -*- coding: utf-8 -*-
import os, re, math, pylab, sys
from math import *
import numpy as np
from StringIO import StringIO
from PIL import Image
from chainer import cuda, Variable, function
from chainer.utils import type_check
from sklearn import preprocessing
import matplotlib.patches as mpatches

def load_images(image_dir, convert_to_grayscale=True, dist="bernoulli"):
	dataset = []
	fs = os.listdir(image_dir)
	print "loading", len(fs), "images..."
	for fn in fs:
		f = open("%s/%s" % (image_dir, fn), "rb")
		if convert_to_grayscale:
			img = np.asarray(Image.open(StringIO(f.read())).convert("L"), dtype=np.float32) / 255.0
		else:
			img = np.asarray(Image.open(StringIO(f.read())).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
		if dist == "bernoulli":
			# Sampling
			img = preprocessing.binarize(img, threshold=0.5)
			pass
		elif dist == "gaussian":
			pass
		else:
			raise Exception()
		dataset.append(img)
		f.close()
	return dataset

def load_labeled_images(image_dir, convert_to_grayscale=True, dist="bernoulli"):
	dataset = []
	labels = []
	fs = os.listdir(image_dir)
	i = 0
	for fn in fs:
		m = re.match("([0-9]+)_.+", fn)
		label = int(m.group(1))
		f = open("%s/%s" % (image_dir, fn), "rb")
		if convert_to_grayscale:
			img = np.asarray(Image.open(StringIO(f.read())).convert("L"), dtype=np.float32) / 255.0
		else:
			img = np.asarray(Image.open(StringIO(f.read())).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
		if dist == "bernoulli":
			# Sampling
			img = preprocessing.binarize(img, threshold=0.5)
			pass
		elif dist == "gaussian":
			pass
		else:
			raise Exception()
		dataset.append(img)
		labels.append(label)
		f.close()
		i += 1
		if i % 100 == 0:
			sys.stdout.write("\rloading images...({:d} / {:d})".format(i, len(fs)))
			sys.stdout.flush()
	sys.stdout.write("\n")
	return dataset, labels

def create_semisupervised(dataset, labels, num_validation_data=10000, num_labeled_data=100, num_types_of_label=10):
	if len(dataset) < num_validation_data + num_labeled_data:
		raise Exception("len(dataset) < num_validation_data + num_labeled_data")
	training_labeled_x = []
	training_unlabeled_x = []
	validation_x = []
	validation_labels = []
	training_labels = []
	indices_for_label = {}
	num_data_per_label = int(num_labeled_data / num_types_of_label)
	num_unlabeled_data = len(dataset) - num_validation_data - num_labeled_data

	indices = np.arange(len(dataset))
	np.random.shuffle(indices)

	def check(index):
		label = labels[index]
		if label not in indices_for_label:
			indices_for_label[label] = []
			return True
		if len(indices_for_label[label]) < num_data_per_label:
			for i in indices_for_label[label]:
				if i == index:
					return False
			return True
		return False

	for n in xrange(len(dataset)):
		index = indices[n]
		if check(index):
			indices_for_label[labels[index]].append(index)
			training_labeled_x.append(dataset[index])
			training_labels.append(labels[index])
		else:
			if len(training_unlabeled_x) < num_unlabeled_data:
				training_unlabeled_x.append(dataset[index])
			else:
				validation_x.append(dataset[index])
				validation_labels.append(labels[index])

	return training_labeled_x, training_labels, training_unlabeled_x, validation_x, validation_labels

def sample_x_variable(batchsize, ndim_x, dataset, gpu_enabled=True):
	x_batch = np.zeros((batchsize, ndim_x), dtype=np.float32)
	indices = np.random.choice(np.arange(len(dataset), dtype=np.int32), size=batchsize, replace=False)
	for j in range(batchsize):
		data_index = indices[j]
		img = dataset[data_index]
		x_batch[j] = img.reshape((ndim_x,))
	x_batch = Variable(x_batch)
	if gpu_enabled:
		x_batch.to_gpu()
	return x_batch

def sample_x_and_label_variables(batchsize, ndim_x, ndim_y, dataset, labels, gpu_enabled=True):
	x_batch = np.zeros((batchsize, ndim_x), dtype=np.float32)
	# one-hot
	y_batch = np.zeros((batchsize, ndim_y), dtype=np.float32)
	# label id
	label_batch = np.zeros((batchsize,), dtype=np.int32)
	indices = np.random.choice(np.arange(len(dataset), dtype=np.int32), size=batchsize, replace=False)
	for j in range(batchsize):
		data_index = indices[j]
		img = dataset[data_index]
		x_batch[j] = img.reshape((ndim_x,))
		y_batch[j, labels[data_index]] = 1
		label_batch[j] = labels[data_index]
	x_batch = Variable(x_batch)
	y_batch = Variable(y_batch)
	label_batch = Variable(label_batch)
	if gpu_enabled:
		x_batch.to_gpu()
		y_batch.to_gpu()
		label_batch.to_gpu()
	return x_batch, y_batch, label_batch

def visualize_x(reconstructed_x_batch, image_width=28, image_height=28, image_channel=1, dir=None):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(16.0, 16.0)
	pylab.clf()
	if image_channel == 1:
		pylab.gray()
	for m in range(100):
		pylab.subplot(10, 10, m + 1)
		if image_channel == 1:
			pylab.imshow(reconstructed_x_batch[m].reshape((image_width, image_height)), interpolation="none")
		elif image_channel == 3:
			pylab.imshow(reconstructed_x_batch[m].reshape((image_channel, image_width, image_height)), interpolation="none")
		pylab.axis("off")
	pylab.savefig("%s/reconstructed_x.png" % dir)

def visualize_z(z_batch, dir=None):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(20.0, 16.0)
	pylab.clf()
	for n in xrange(z_batch.shape[0]):
		result = pylab.scatter(z_batch[n, 0], z_batch[n, 1], s=40, marker="o", edgecolors='none')
	pylab.xlabel("z1")
	pylab.ylabel("z2")
	pylab.savefig("%s/latent_code.png" % dir)

def visualize_labeled_z(z_batch, label_batch, dir=None):
	fig = pylab.gcf()
	fig.set_size_inches(20.0, 16.0)
	pylab.clf()
	colors = ["#2103c8", "#0e960e", "#e40402","#05aaa8","#ac02ab","#aba808","#151515","#94a169", "#bec9cd", "#6a6551"]
	for n in xrange(z_batch.shape[0]):
		result = pylab.scatter(z_batch[n, 0], z_batch[n, 1], c=colors[label_batch[n]], s=40, marker="o", edgecolors='none')

	classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
	recs = []
	for i in range(0, len(colors)):
		recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=colors[i]))

	ax = pylab.subplot(111)
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax.legend(recs, classes, loc="center left", bbox_to_anchor=(1.1, 0.5))
	pylab.xticks(pylab.arange(-4, 5))
	pylab.yticks(pylab.arange(-4, 5))
	pylab.xlabel("z1")
	pylab.ylabel("z2")
	pylab.savefig("%s/labeled_z.png" % dir)
