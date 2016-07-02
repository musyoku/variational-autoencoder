# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from chainer import cuda, Variable
import pandas as pd
sys.path.append(os.path.split(os.getcwd())[0])
import util
from args import args
from model import conf, vae
from vae_m2 import GaussianM2VAE

dist = "bernoulli"
if isinstance(vae, GaussianM2VAE):
	dist = "gaussian"
dataset, labels = util.load_labeled_images(args.train_image_dir, dist=dist)

max_epoch = 1000
num_trains_per_epoch = 2000
batchsize_l = 100
batchsize_u = 100

# Create labeled/unlabeled split in training set
num_types_of_label = 10
num_labeled_data = args.num_labeled_data
num_validation_data = 10000
labeled_dataset, labels, unlabeled_dataset, validation_dataset, validation_labels = util.create_semisupervised(dataset, labels, num_validation_data, num_labeled_data, num_types_of_label)
print "labels:", labels
alpha = 0.1 * len(dataset) / len(labeled_dataset)
alpha = 1.0
print "alpha:", alpha
print "dataset:: labeled: {:d} unlabeled: {:d} validation: {:d}".format(len(labeled_dataset), len(unlabeled_dataset), len(validation_dataset))

if num_labeled_data < batchsize_l:
	batchsize_l = num_labeled_data
	
if len(unlabeled_dataset) < batchsize_u:
	batchsize_u = len(unlabeled_dataset)

# from PIL import Image
# for i in xrange(len(labeled_dataset)):
# 	image = Image.fromarray(np.uint8(labeled_dataset[i].reshape(28, 28) * 255))
# 	image.save("labeled_images/{:d}.bmp".format(i))

# Export result to csv
csv_epoch = []

total_time = 0
for epoch in xrange(max_epoch):
	sum_loss_labeled = 0
	sum_loss_unlabeled = 0
	sum_loss_classifier = 0
	epoch_time = time.time()
	for t in xrange(num_trains_per_epoch):
		x_labeled, y_labeled, label_ids = util.sample_x_and_label_variables(batchsize_l, conf.ndim_x, conf.ndim_y, labeled_dataset, labels, gpu_enabled=conf.gpu_enabled)
		x_unlabeled = util.sample_x_variable(batchsize_u, conf.ndim_x, unlabeled_dataset, gpu_enabled=conf.gpu_enabled)

		# train
		loss_labeled, loss_unlabeled = vae.train(x_labeled, y_labeled, label_ids, x_unlabeled)
		loss_classifier = vae.train_classification(x_labeled, label_ids, alpha=alpha)

		sum_loss_labeled += loss_labeled
		sum_loss_unlabeled += loss_unlabeled
		sum_loss_classifier += loss_classifier
		if t % 10 == 0:
			sys.stdout.write("\rTraining in progress...({:d} / {:d})".format(t, num_trains_per_epoch))
			sys.stdout.flush()
	epoch_time = time.time() - epoch_time
	total_time += epoch_time
	sys.stdout.write("\r")
	print "epoch: {:d} loss:: labeled: {:.3f} unlabeled: {:.3f} classifier: {:.3f} time: {:d} min total: {:d} min".format(epoch + 1, sum_loss_labeled / num_trains_per_epoch, sum_loss_unlabeled / num_trains_per_epoch, sum_loss_classifier / num_trains_per_epoch, int(epoch_time / 60), int(total_time / 60))
	sys.stdout.flush()
	vae.save(args.model_dir)

	# validation
	x_labeled, _, label_ids = util.sample_x_and_label_variables(num_validation_data, conf.ndim_x, conf.ndim_y, validation_dataset, validation_labels, gpu_enabled=False)
	if conf.gpu_enabled:
		x_labeled.to_gpu()
	prediction = vae.sample_x_label(x_labeled, test=True, argmax=True)
	correct = 0
	for i in xrange(num_validation_data):
		if prediction[i] == label_ids.data[i]:
			correct += 1
	print "validation:: classification accuracy: {:f}".format(correct / float(num_validation_data))

	# Export to csv
	csv_epoch.append([epoch, int(total_time / 60), correct / float(num_validation_data)])
	if epoch % 10 == 0 and len(csv_epoch) > 0:
		data = pd.DataFrame(csv_epoch)
		data.columns = ["epoch", "min", "accuracy"]
		data.to_csv("{:s}/epoch.csv".format(args.model_dir))

