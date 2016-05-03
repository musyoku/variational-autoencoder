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

def validate_x_label():
	num_validation = 1000
	x_labeled, y_labeled = util.sample_x_and_y_variables(num_validation, conf.ndim_x, conf.ndim_y, dataset, labels, one_hot_label=False, use_gpu=False)
	if conf.use_gpu:
		x_labeled.to_gpu()
	prediction = vae.sample_x_label(x_labeled, test=True)

	correct = 0
	for i in xrange(num_validation):
		if prediction[i] == y_labeled.data[i]:
			correct += 1

	print "Classification accuracy:", correct / float(num_validation)

validate_x_label()
