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
from PIL import Image

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
print x.data[0]
mean, ln_var = vae.encoder(x, test=True, sample_output=False)
# print ln_var.data
z = F.gaussian(mean, ln_var)
# print cuda.cupy.amax(z.data)
# i = cuda.cupy.argmax(cuda.cupy.amax(z.data, axis=1))
# image = Image.fromarray(np.uint8((cuda.to_cpu(x.data)[i].reshape(28, 28) + 1.0) / 2.0 * 255.0))
# image.save("unko.bmp")
# print mean.data
# print F.exp(ln_var).data
mean, ln_var = vae.decoder(z, test=True, output_pixel_expectation=False)
print mean.data[0]
print F.exp(ln_var).data[0]
_x = F.gaussian(mean, ln_var)
print _x.data[0]
if conf.gpu_enabled:
	z.to_cpu()
	_x.to_cpu()
util.visualize_x(_x.data, dir=args.vis_dir)
util.visualize_z(z.data, dir=args.vis_dir)
util.visualize_labeled_z(z.data, label_ids.data, dir=args.vis_dir)