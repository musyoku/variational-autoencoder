# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from chainer import cuda, Variable
sys.path.append(os.path.split(os.getcwd())[0])
import util
from args import args
from model import conf1, vae1, conf2, vae2
from vae_m1 import GaussianM1VAE

dist = "bernoulli"
if isinstance(vae1, GaussianM1VAE):
	dist = "gaussian"
dataset, labels = util.load_labeled_images(args.train_image_dir, dist=dist)

max_epoch = 1000
vae1_num_trains_per_epoch = 5000
vae2_num_trains_per_epoch = 2000
batchsize = 100

# Create labeled/unlabeled split in training set
num_types_of_label = 10
num_labeled_data = 100
if num_labeled_data < batchsize:
	batchsize = num_labeled_data
num_validation_data = 10000
labeled_dataset, labels, unlabeled_dataset, validation_dataset, validation_labels = util.create_semisupervised(dataset, labels, num_validation_data, num_labeled_data, num_types_of_label)
print "labels:", labels
alpha = 0.1 * len(dataset) / len(labeled_dataset)
print "alpha:", alpha
print "dataset:: labeled: {:d} unlabeled: {:d} validation: {:d}".format(len(labeled_dataset), len(unlabeled_dataset), len(validation_dataset))

total_time = 0
for epoch in xrange(max_epoch):
	# Train M1
	sum_loss = 0
	epoch_time = time.time()
	for t in xrange(vae1_num_trains_per_epoch):
		x = util.sample_x_variable(batchsize, conf1.ndim_x, dataset, gpu_enabled=conf1.gpu_enabled)

		# train
		loss = vae1.train(x, L=1)

		sum_loss += loss
		if t % 10 == 0:
			sys.stdout.write("\rTraining M1 in progress...(%d / %d)" % (t, vae1_num_trains_per_epoch))
			sys.stdout.flush()
	epoch_time = time.time() - epoch_time
	total_time += epoch_time
	sys.stdout.write("\r")
	print "[M1] epoch:", epoch, "loss: {:.3f}".format(sum_loss / vae1_num_trains_per_epoch), "time: {:d} min".format(int(epoch_time / 60)), "total: {:d} min".format(int(total_time / 60))
	sys.stdout.flush()
	vae1.save(args.model_dir)


	# Train M2
	sum_loss_labeled = 0
	sum_loss_unlabeled = 0
	sum_loss_classifier = 0
	epoch_time = time.time()
	for t in xrange(vae2_num_trains_per_epoch):
		x_labeled, y_labeled, label_ids = util.sample_x_and_label_variables(batchsize, conf1.ndim_x, conf2.ndim_y, labeled_dataset, labels, gpu_enabled=conf2.gpu_enabled)
		x_unlabeled = util.sample_x_variable(batchsize, conf1.ndim_x, unlabeled_dataset, gpu_enabled=conf2.gpu_enabled)
		z_labeled = Variable(vae1.encoder(x_labeled, test=True, sample_output=True).data)
		z_unlabeled = Variable(vae1.encoder(x_unlabeled, test=True, sample_output=True).data)

		# train
		loss_labeled, loss_unlabeled = vae2.train(z_labeled, y_labeled, label_ids, z_unlabeled)
		loss_classifier = vae2.train_classification(z_labeled, label_ids, alpha=alpha)
		
		sum_loss_labeled += loss_labeled
		sum_loss_unlabeled += loss_unlabeled
		sum_loss_classifier += loss_classifier
		if t % 10 == 0:
			sys.stdout.write("\rTraining M2 in progress...({:d} / {:d})".format(t, vae2_num_trains_per_epoch))
			sys.stdout.flush()
	epoch_time = time.time() - epoch_time
	total_time += epoch_time
	sys.stdout.write("\r")
	print "[M2] epoch:", epoch, "loss::", "labeled: {:.3f}".format(sum_loss_labeled / vae2_num_trains_per_epoch), "unlabeled: {:.3f}".format(sum_loss_unlabeled / vae2_num_trains_per_epoch), "classifier: {:.3f}".format(sum_loss_classifier / vae2_num_trains_per_epoch), "time: {:d} min".format(int(epoch_time / 60)), "total: {:d} min".format(int(total_time / 60))
	sys.stdout.flush()
	vae2.save(args.model_dir)

	# validation
	x_labeled, _, label_ids = util.sample_x_and_label_variables(num_validation_data, conf1.ndim_x, conf2.ndim_y, validation_dataset, validation_labels, gpu_enabled=False)
	if conf1.gpu_enabled:
		x_labeled.to_gpu()
	z_labeled = vae1.encoder(x_labeled, test=True)
	prediction = vae2.sample_x_label(z_labeled, test=True, argmax=True)
	correct = 0
	for i in xrange(num_validation_data):
		if prediction[i] == label_ids.data[i]:
			correct += 1
	print "validation:: classification accuracy: {:f}".format(correct / float(num_validation_data))

