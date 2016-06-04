# -*- coding: utf-8 -*-
import os, sys, time, pylab
import numpy as np
from chainer import cuda, Variable
import matplotlib.patches as mpatches
sys.path.append(os.path.split(os.getcwd())[0])
import util
from args import args
from model import conf, vae
from vae_m2 import GaussianM2VAE

try:
	os.mkdir(args.vis_dir)
except:
	pass
dist = "bernoulli"
if isinstance(vae, GaussianM2VAE):
	dist = "gaussian"
dataset, labels = util.load_labeled_images(args.test_image_dir, dist=dist)

def forward_one_step(num_images):
	x, y_labeled, label_ids = util.sample_x_and_label_variables(num_images, conf.ndim_x, conf.ndim_y, dataset, labels, gpu_enabled=False)
	x.to_gpu()
	y = vae.sample_x_y(x, test=True)
	z = vae.encoder_xy_z(x, y, test=True)
	_x = vae.decode_zy_x(z, y, test=True)
	if conf.gpu_enabled:
		z.to_cpu()
		_x.to_cpu()
	_x = _x.data
	return z, _x, label_ids

z, _x, _ = forward_one_step(100)
util.visualize_x(_x, dir=args.vis_dir)

z, _x, label_ids = forward_one_step(5000)
util.visualize_z(z.data, dir=args.vis_dir)
util.visualize_labeled_z(z.data, label_ids.data, dir=args.vis_dir)