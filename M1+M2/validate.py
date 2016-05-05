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
dataset, labels = util.load_labeled_images(args.test_image_dir)

def validate_x_label():
	num_validation = 1000
	x_labeled, _, label_ids = util.sample_x_and_label_variables(num_validation, conf1.ndim_x, conf2.ndim_y, dataset, labels, use_gpu=False)
	if conf1.use_gpu:
		x_labeled.to_gpu()
	z_labeled = vae1.encode(x_labeled, test=True)
	prediction = vae2.sample_x_label(z_labeled, test=True, argmax=True)

	correct = 0
	for i in xrange(num_validation):
		if prediction[i] == label_ids.data[i]:
			correct += 1

	print "Classification accuracy:", correct / float(num_validation)

validate_x_label()
