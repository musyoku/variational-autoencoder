# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from chainer import cuda, Variable
sys.path.append(os.path.split(os.getcwd())[0])
import util
from args import args
from model import conf, vae

vae.load(args.model_dir)
dataset, labels = util.load_labeled_images(args.train_image_dir)

max_epoch = 1000
num_trains_per_epoch = 1000
batchsize = 100

# Create labeled/unlabeled split in training set
num_labbeled_data = 300
num_validation_data = 10000
labeled_dataset, labels, unlabeled_dataset, validation_dataset, validation_labels = util.create_semisupervised(dataset, labels, num_validation_data, num_labbeled_data)
print "labels:", labels
alpha = 0.1 * len(dataset) / len(labeled_dataset)
print "alpha:", alpha
print "dataset:: labeled: {:d} unlabeled: {:d} validation: {:d}".format(len(labeled_dataset), len(unlabeled_dataset), len(validation_dataset))

total_time = 0
for epoch in xrange(max_epoch):
	sum_loss_labeled = 0
	sum_loss_unlabeled = 0
	sum_loss_classifier = 0
	epoch_time = time.time()
	for t in xrange(num_trains_per_epoch):
		x_labeled, y_labeled, label_ids = util.sample_x_and_label_variables(batchsize, conf.ndim_x, conf.ndim_y, labeled_dataset, labels, use_gpu=conf.use_gpu)
		x_unlabeled = util.sample_x_variable(batchsize, conf.ndim_x, unlabeled_dataset, use_gpu=conf.use_gpu)

		# train
		loss_labeled, loss_unlabeled = vae.train(x_labeled, y_labeled, label_ids, x_unlabeled)
		loss_classifier = vae.train_classification(x_labeled, label_ids, alpha=alpha)

		sum_loss_labeled += loss_labeled
		sum_loss_unlabeled += loss_unlabeled
		sum_loss_classifier += loss_classifier
		if t % 100 == 0:
			sys.stdout.write("\rTraining in progress...({:d} / {:d})".format(t, num_trains_per_epoch))
			sys.stdout.flush()
	epoch_time = time.time() - epoch_time
	total_time += epoch_time
	sys.stdout.write("\r")
	print "epoch: {:d} loss:: labeled: {:.3f} unlabeled: {:.3f} classifier: {:.3f} time: {:d} min total: {:d} min".format(epoch + 1, sum_loss_labeled / num_trains_per_epoch, sum_loss_unlabeled / num_trains_per_epoch, sum_loss_classifier / num_trains_per_epoch, int(epoch_time / 60), int(total_time / 60))
	sys.stdout.flush()
	vae.save(args.model_dir)

	# validation
	x_labeled, _, label_ids = util.sample_x_and_label_variables(num_validation_data, conf.ndim_x, conf.ndim_y, validation_dataset, validation_labels, sequential=True, use_gpu=False)
	if conf.use_gpu:
		x_labeled.to_gpu()
	prediction = vae.sample_x_label(x_labeled, test=True, argmax=True)
	correct = 0
	for i in xrange(num_validation_data):
		if prediction[i] == label_ids.data[i]:
			correct += 1
	print "validation:: classification accuracy: {:f}".format(correct / float(num_validation_data))

