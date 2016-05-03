# -*- coding: utf-8 -*-
import sys, os, math
import numpy as np
from chainer import Variable, cuda
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
from vae_m2 import BernoulliM2VAE, Conf

conf = Conf()
conf.ndim_y = 2
conf.ndim_x = 5
vae = BernoulliM2VAE(conf, name="m2")

batchsize = 1
y = np.zeros((batchsize, conf.ndim_y), dtype=np.int32)
y[0, 1] = 1
y = Variable(y)
y.to_gpu()

x = np.random.normal(size=(batchsize, conf.ndim_x)).astype(np.float32)
x = Variable(x)
x.to_gpu()

xp = cuda.cupy

for i in xrange(1000):
	y_distribution = vae.encode_x_y(x)
	print y_distribution.data
	mask = y.data.astype(xp.float32)
	mask = Variable(mask)
	loss = F.sum(mask * -F.log(y_distribution + 1e-6)) / batchsize
	vae.zero_grads()
	loss.backward()
	vae.update()

