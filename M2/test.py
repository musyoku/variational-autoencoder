# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from chainer import cuda, Variable
sys.path.append(os.path.split(os.getcwd())[0])
import util
from args import args
from model import conf, vae
from vae_m2 import GaussianM2VAE

dist = "bernoulli"
if isinstance(vae, GaussianM2VAE):
	dist = "gaussian"
dataset, labels = util.load_labeled_images(args.test_image_dir, dist=dist)
num_data = len(dataset)

x_labeled, _, label_ids = util.sample_x_and_label_variables(num_data, conf.ndim_x, conf.ndim_y, dataset, labels, gpu_enabled=False)
if conf.gpu_enabled:
	x_labeled.to_gpu()
prediction = vae.sample_x_label(x_labeled, test=True, argmax=True)
correct = 0
for i in xrange(num_data):
	if prediction[i] == label_ids.data[i]:
		correct += 1
print "test:: classification accuracy: {:f}".format(correct / float(num_data))