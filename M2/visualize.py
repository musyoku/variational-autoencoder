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
dataset = util.load_images(args.test_image_dir, dist=dist)

num_images = 100
x = util.sample_x_variable(num_images, conf.ndim_x, dataset, gpu_enabled=conf.gpu_enabled)
y = vae.sample_x_y(x, test=True)
z = vae.encoder_xy_z(x, y, test=True)
_x = vae.decode_zy_x(z, y, test=True)
if conf.gpu_enabled:
	z.to_cpu()
	_x.to_cpu()
_x = _x.data
if dist == "gaussian":
	# [-1,1] => [0,1]
	_x = (_x + 1.0) / 2.0
util.visualize_x(_x, dir=args.vis_dir)
util.visualize_z(z.data, dir=args.vis_dir)