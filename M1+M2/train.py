# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from chainer import cuda, Variable
sys.path.append(os.path.split(os.getcwd())[0])
import util
from args import args
from model import conf1, vae1, conf2, vae2

vae1.load(args.model_dir)
vae2.load(args.model_dir)
dataset, labels = util.load_labeled_images(args.train_image_dir)

max_epoch = 1000
num_trains_per_epoch = 1000
batchsize = 100

# Create labeled/unlabeled split in training set
max_labbeled_data = 100
labeled_dataset, labels, unlabeled_dataset = util.create_semisupervised(dataset, labels, max_labbeled_data)
print "labels:", labels
alpha = 0.1 * len(dataset) / len(labeled_dataset)
print "alpha:", alpha
print "dataset::", "labeled:", len(labeled_dataset), "unlabeled:", len(unlabeled_dataset)

total_time = 0
for epoch in xrange(max_epoch):
	# Train M1
	sum_loss = 0
	epoch_time = time.time()
	for t in xrange(num_trains_per_epoch):
		x = util.sample_x_variable(batchsize, conf1.ndim_x, dataset, use_gpu=conf1.use_gpu)
		loss = vae1.train(x)
		sum_loss += loss
		if t % 100 == 0:
			sys.stdout.write("\rTraining M1 in progress...(%d / %d)" % (t, num_trains_per_epoch))
			sys.stdout.flush()
	epoch_time = time.time() - epoch_time
	total_time += epoch_time
	sys.stdout.write("\r")
	print "[M1] epoch:", epoch, "loss: {:.3f}".format(sum_loss / num_trains_per_epoch), "time: {:d}min".format(int(epoch_time / 60)), "total: {:d}min".format(int(total_time / 60))
	sys.stdout.flush()
	vae1.save(args.model_dir)

	# Train M2
	sum_loss_labeled = 0
	sum_loss_unlabeled = 0
	sum_loss_classifier = 0
	epoch_time = time.time()
	for t in xrange(num_trains_per_epoch):
		x_labeled, y_labeled, label_ids = util.sample_x_and_label_variables(batchsize, conf1.ndim_x, conf2.ndim_y, labeled_dataset, labels, use_gpu=conf2.use_gpu)
		x_unlabeled = util.sample_x_variable(batchsize, conf1.ndim_x, unlabeled_dataset, use_gpu=conf2.use_gpu)
		z_labeled = vae1.encode(x_labeled, test=True)
		z_unlabeled = vae1.encode(x_unlabeled, test=True)
		loss_labeled, loss_unlabeled, loss_classifier = vae2.train(z_labeled, y_labeled, label_ids, z_unlabeled, alpha)
		sum_loss_labeled += loss_labeled
		sum_loss_unlabeled += loss_unlabeled
		sum_loss_classifier += loss_classifier
		if t % 100 == 0:
			sys.stdout.write("\rTraining M2 in progress...({:d} / {:d})".format(t, num_trains_per_epoch))
			sys.stdout.flush()
	epoch_time = time.time() - epoch_time
	total_time += epoch_time
	sys.stdout.write("\r")
	print "[M2] epoch:", epoch, "loss::", "labeled: {:.3f}".format(sum_loss_labeled / num_trains_per_epoch), "unlabeled: {:.3f}".format(sum_loss_unlabeled / num_trains_per_epoch), "classifier: {:.3f}".format(sum_loss_classifier / num_trains_per_epoch), "time: {:d} min".format(int(epoch_time / 60)), "total: {:d} min".format(int(total_time / 60))
	sys.stdout.flush()
	vae2.save(args.model_dir)