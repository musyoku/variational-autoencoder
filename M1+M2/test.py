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
dataset, labels = util.load_labeled_images(args.test_image_dir, dist=dist)
num_data = len(dataset)

x_labeled, _, label_ids = util.sample_x_and_label_variables(num_data, conf1.ndim_x, conf2.ndim_y, dataset, labels, gpu_enabled=False)
if conf1.gpu_enabled:
	x_labeled.to_gpu()
z_labeled = vae1.encoder(x_labeled, test=True)
prediction = vae2.sample_x_label(z_labeled, test=True, argmax=True)

correct = 0
for i in xrange(num_data):
	if prediction[i] == label_ids.data[i]:
		correct += 1

print "test:: classification accuracy: {:.3f}".format(correct / float(num_data))

