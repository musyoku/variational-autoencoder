# -*- coding: utf-8 -*-
import numpy as np
import os, sys
from chainer import cuda, Variable
sys.path.append(os.path.split(os.getcwd())[0])
from vae_m2 import BernoulliM2VAE, GaussianM2VAE, Conf

conf = Conf()
conf.gpu_enabled = False
conf.ndim_x = 5
conf.ndim_y = 2
conf.ndim_z = 4
conf.encoder_xy_z_apply_dropout = False
conf.encoder_x_y_apply_dropout = False
conf.decoder_apply_dropout = False
conf.encoder_apply_batchnorm = False
conf.encoder_apply_batchnorm_to_input = False
conf.decoder_apply_batchnorm = False
conf.decoder_apply_batchnorm_to_input = False
conf.encoder_xy_z_hidden_units = [50]
conf.encoder_x_y_hidden_units = [50]
conf.decoder_hidden_units = [50]
conf.encoder_xy_z_apply_batchnorm = False
conf.encoder_xy_z_apply_batchnorm_to_input = False
conf.encoder_x_y_apply_batchnorm = False
conf.encoder_x_y_apply_batchnorm_to_input = False
conf.decoder_apply_batchnorm = False
conf.decoder_apply_batchnorm_to_input = False
vae = GaussianM2VAE(conf, name="m1")

x_labeled = np.asarray([
	[1, -1, 1, -1, 1],
	[-1, 1, -1, 1, -1],
], dtype=np.float32)
x_labeled = Variable(x_labeled)
x_unlabeled = np.asarray([
	[1, 0, -1, 0, 1],
	[0, 1, 0, 1, -1],
], dtype=np.float32)
x_unlabeled = Variable(x_unlabeled)
y_labeled = np.asarray([
	[0, 1],
	[1, 0],
], dtype=np.float32)
y_labeled = Variable(y_labeled)
label_ids = np.asarray([
	0, 1
], dtype=np.int32)
label_ids = Variable(label_ids)
if conf.gpu_enabled:
	x_labeled.to_gpu()
	x_unlabeled.to_gpu()
	y_labeled.to_gpu()
	label_ids.to_gpu()
sum_loss_labeled = 0
sum_loss_unlabeled = 0
sum_loss_classifier = 0
for i in xrange(10000):
	loss_labeled, loss_unlabeled = vae.train(x_labeled, y_labeled, label_ids, x_unlabeled)
	loss_classifier = vae.train_classification(x_labeled, label_ids, alpha=1)
	sum_loss_labeled += loss_labeled
	sum_loss_unlabeled += loss_unlabeled
	sum_loss_classifier += loss_classifier
	if i % 100 == 0 and i > 0:
		print sum_loss_labeled / 100.0, sum_loss_unlabeled / 100.0, sum_loss_classifier / 100.0
		sum_loss_labeled = 0
		sum_loss_unlabeled = 0
		sum_loss_classifier = 0
		z = vae.encode_x_z(x_labeled, test=True)
		x = vae.decode_zy_x(z, y_labeled, test=True)
		print x.data
		z = vae.encode_x_z(x_unlabeled, test=True)
		y = vae.sample_x_y(x_unlabeled, test=True)
		x = vae.decode_zy_x(z, y, test=True)
		print x.data
