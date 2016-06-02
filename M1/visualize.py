# -*- coding: utf-8 -*-
import os, sys, time, pylab
import numpy as np
from chainer import cuda, Variable
import matplotlib.patches as mpatches
sys.path.append(os.path.split(os.getcwd())[0])
import util
from args import args
from model import conf, vae
from vae_m1 import GaussianM1VAE
from chainer import functions as F

try:
	os.mkdir(args.vis_dir)
except:
	pass

dist = "bernoulli"
if isinstance(vae, GaussianM1VAE):
	dist = "gaussian"
dataset, labels = util.load_labeled_images(args.train_image_dir, dist=dist)

num_images = 5000
x, y_labeled, label_ids = util.sample_x_and_label_variables(num_images, conf.ndim_x, 10, dataset, labels, gpu_enabled=False)
if conf.gpu_enabled:
	x.to_gpu()
z = vae.encoder(x, test=True)
mean, ln_var = vae.decoder(z, test=True, output_pixel_value=False)
_x = vae.decoder(z, test=True, output_pixel_value=True)
if conf.gpu_enabled:
	z.to_cpu()
	_x.to_cpu()
util.visualize_gaussian_x(_x.data, dir=args.vis_dir)
util.visualize_z(z.data, dir=args.vis_dir)
util.visualize_labeled_z(z.data, label_ids.data, dir=args.vis_dir)