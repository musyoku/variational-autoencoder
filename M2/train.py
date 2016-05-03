# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from chainer import cuda, Variable
sys.path.append(os.path.split(os.getcwd())[0])
import util
from args import args
from model import conf, vae

vae.load(args.model_dir)
dataset, labels = util.load_labeled_images(args.image_dir)

max_epoch = 1000
num_trains_per_epoch = 1000
batchsize = 128

# Create labeled/unlabeled split in training set
max_labbeled_data = 100
labeled_dataset, labels, unlabeled_dataset = util.create_semisupervised(dataset, labels, max_labbeled_data)
print "labels:", labels
alpha = 0.1 * len(dataset) / conf.ndim_y / len(labeled_dataset)
print "alpha:", alpha
print "dataset::", "labeled:", len(labeled_dataset), "unlabeled:", len(unlabeled_dataset)

def train_supervised():
	total_time = 0
	for epoch in xrange(max_epoch):
		sum_loss = 0
		epoch_time = time.time()
		for t in xrange(num_trains_per_epoch):
			x_labeled, y_labeled = util.sample_x_and_y_variables(batchsize, conf.ndim_x, conf.ndim_y, labeled_dataset, labels, use_gpu=conf.use_gpu)
			loss = vae.train_supervised(x_labeled, y_labeled, alpha)
			sum_loss += loss
			if t % 100 == 0:
				sys.stdout.write("\rTraining in progress...(%d / %d)" % (t, num_trains_per_epoch))
				sys.stdout.flush()
		epoch_time = time.time() - epoch_time
		total_time += epoch_time
		sys.stdout.write("\r")
		print "epoch:", epoch, "loss::", "labeled:", sum_loss / num_trains_per_epoch, "time:", "%d" % (epoch_time / 60), "min", "total", "%d" % (total_time / 60), "min"
		sys.stdout.flush()
		vae.save(args.model_dir)

def train_semisupervised():
	total_time = 0
	for epoch in xrange(max_epoch):
		sum_loss_labeled = 0
		sum_loss_unlabeled = 0
		epoch_time = time.time()
		for t in xrange(num_trains_per_epoch):
			x_labeled, y_labeled = util.sample_x_and_y_variables(batchsize, conf.ndim_x, conf.ndim_y, labeled_dataset, labels, use_gpu=conf.use_gpu)
			x_unlabeled = util.sample_x_variable(batchsize, conf.ndim_x, unlabeled_dataset, use_gpu=conf.use_gpu)
			loss_labeled, loss_unlabeled = vae.train(x_labeled, y_labeled, x_unlabeled, alpha)
			sum_loss_labeled += loss_labeled
			sum_loss_unlabeled += loss_unlabeled
			if t % 100 == 0:
				sys.stdout.write("\rTraining in progress...(%d / %d)" % (t, num_trains_per_epoch))
				sys.stdout.flush()
		epoch_time = time.time() - epoch_time
		total_time += epoch_time
		sys.stdout.write("\r")
		print "epoch:", epoch, "loss::", "labeled:", sum_loss_labeled / num_trains_per_epoch, "unlabeled:", sum_loss_unlabeled / num_trains_per_epoch, "time:", "%d" % (epoch_time / 60), "min", "total", "%d" % (total_time / 60), "min"
		sys.stdout.flush()
		vae.save(args.model_dir)

train_semisupervised()
