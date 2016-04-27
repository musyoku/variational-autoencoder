# -*- coding: utf-8 -*-
import os, re, math, pylab
from math import *
import numpy as np
from StringIO import StringIO
from PIL import Image
from chainer import cuda, Variable, function
from chainer.utils import type_check

def load_images(args, convert_to_grayscale=True):
	dataset = []
	fs = os.listdir(args.image_dir)
	print "loading", len(fs), "images..."
	for fn in fs:
		f = open("%s/%s" % (args.image_dir, fn), "rb")
		if convert_to_grayscale:
			img = np.asarray(Image.open(StringIO(f.read())).convert("L"), dtype=np.float32) / 255.0
		else:
			img = np.asarray(Image.open(StringIO(f.read())).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
		img = (img - 0.5) / 0.5
		dataset.append(img)
		f.close()
	return dataset

def load_labeled_dataset(args, convert_to_grayscale=True):
	dataset = []
	labels = []
	fs = os.listdir(args.image_dir)
	print "loading", len(fs), "images..."
	for fn in fs:
		m = re.match("(.)_.+", fn)
		label = int(m.group(1))
		f = open("%s/%s" % (args.image_dir, fn), "rb")
		if convert_to_grayscale:
			img = np.asarray(Image.open(StringIO(f.read())).convert("L"), dtype=np.float32) / 255.0
		else:
			img = np.asarray(Image.open(StringIO(f.read())).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
		img = (img - 0.5) / 0.5
		dataset.append(img)
		labels.append(label)
		f.close()
	return dataset, labels

class Concat(function.Function):
	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(n_in == 2)
		x_type, label_type = in_types

		type_check.expect(
			x_type.dtype == np.float32,
			label_type.dtype == np.float32,
			x_type.ndim == 2,
			label_type.ndim == 2,
		)

	def forward(self, inputs):
		xp = cuda.get_array_module(inputs[0])
		x, label = inputs
		n_batch = x.shape[0]
		output = xp.empty((n_batch, x.shape[1] + label.shape[1]), dtype=xp.float32)
		output[:,:x.shape[1]] = x
		output[:,x.shape[1]:] = label
		return output,

	def backward(self, inputs, grad_outputs):
		x, label = inputs
		return grad_outputs[0][:,:x.shape[1]], grad_outputs[0][:,x.shape[1]:]

def append_label(x, label):
	return Concat()(x, label)