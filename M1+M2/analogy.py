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

try:
	os.mkdir(args.vis_dir)
except:
	pass

n_analogies = 10
n_image_channels = 1
image_width = 28
image_height = 28
x = util.sample_x_variable(10, conf1.ndim_x, dataset, gpu_enabled=conf1.gpu_enabled)
z1 = vae1.encode(x, test=True)
y = vae2.encode_x_y(z1, test=True)
z2 = vae2.encode_xy_z(z1, y, test=True)

fig = pylab.gcf()
fig.set_size_inches(16.0, 16.0)
pylab.clf()
if n_image_channels == 1:
	pylab.gray()
xp = np
if conf1.gpu_enabled:
	x.to_cpu()
	xp = cuda.cupy
for m in xrange(n_analogies):
	pylab.subplot(n_analogies, conf2.ndim_y + 2, m * 12 + 1)
	if n_image_channels == 1:
		pylab.imshow(np.clip((x.data[m] + 1.0) / 2.0, 0.0, 1.0).reshape((image_width, image_height)), interpolation="none")
	elif n_image_channels == 3:
		pylab.imshow(np.clip((x.data[m] + 1.0) / 2.0, 0.0, 1.0).reshape((n_image_channels, image_width, image_height)), interpolation="none")
	pylab.axis("off")
for m in xrange(n_analogies):
	analogy_y = xp.identity(conf2.ndim_y, dtype=xp.float32)
	analogy_y = Variable(analogy_y)
	base_z2 = xp.empty((n_analogies, z2.data.shape[1]), dtype=xp.float32)
	for n in xrange(conf2.ndim_y):
		base_z2[n] = z2.data[m]
	base_z2 = Variable(base_z2)
	_z1 = vae2.decode_zy_x(base_z2, analogy_y, test=True, output_pixel_value=True)
	_x = vae1.decode(_z1, test=True)
	if conf1.gpu_enabled:
		_x.to_cpu()
	for n in xrange(conf2.ndim_y):
		pylab.subplot(n_analogies, conf2.ndim_y + 2, m * 12 + 3 + n)
		if n_image_channels == 1:
			pylab.imshow(np.clip((_x.data[n] + 1.0) / 2.0, 0.0, 1.0).reshape((image_width, image_height)), interpolation="none")
		elif n_image_channels == 3:
			pylab.imshow(np.clip((_x.data[n] + 1.0) / 2.0, 0.0, 1.0).reshape((n_image_channels, image_width, image_height)), interpolation="none")
		pylab.axis("off")

pylab.savefig("{:s}/analogy.png".format(args.vis_dir))

