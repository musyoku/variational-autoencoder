# -*- coding: utf-8 -*-
import os, sys, time, pylab
import numpy as np
from chainer import cuda, Variable
import matplotlib.patches as mpatches
sys.path.append(os.path.split(os.getcwd())[0])
import util
from args import args
from model import conf1, vae1, conf2, vae2

try:
	os.mkdir(args.vis_dir)
except:
	pass

vae1.load(args.model_dir)
vae2.load(args.model_dir)
dataset = util.load_images(args.test_image_dir)

num_images = 100
x = util.sample_x_variable(num_images, conf1.ndim_x, dataset, gpu_enabled=conf1.gpu_enabled)
z1 = vae1.encode(x, test=True)
y = vae2.encode_x_y(z1, test=True)
z2 = vae2.encode_xy_z(z1, y, test=True)
_z1 = vae2.decode_zy_x(z2, y, test=True, output_pixel_value=True)
_x = vae1.decode(_z1, test=True)
if conf1.gpu_enabled:
	z2.to_cpu()
	_x.to_cpu()
util.visualize_x(_x.data, dir=args.vis_dir)
util.visualize_z(z2.data, dir=args.vis_dir)