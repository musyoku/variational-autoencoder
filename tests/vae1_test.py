# -*- coding: utf-8 -*-
import numpy as np
import os, sys
from chainer import cuda, Variable
sys.path.append(os.path.split(os.getcwd())[0])
from vae_m1 import BernoulliM1VAE, GaussianM1VAE, Conf

conf = Conf()
conf.gpu_enabled = True
conf.ndim_z = 4
conf.ndim_x = 5
conf.encoder_apply_dropout = False
conf.decoder_apply_dropout = False
conf.encoder_apply_batchnorm = False
conf.encoder_apply_batchnorm_to_input = False
conf.decoder_apply_batchnorm = False
conf.decoder_apply_batchnorm_to_input = False
conf.encoder_units = [10, 10]
conf.decoder_units = [10, 10]
vae = GaussianM1VAE(conf, name="m1")

x = np.asarray([
	[1, -1, 1, -1, 1],
	[-1, 1, -1, 1, -1],
], dtype=np.float32)
x = Variable(x)
x.to_gpu()
sum_los = 0
for i in xrange(10000):
	sum_los += vae.train(x)
	if i % 100 == 0 and i > 0:
		print sum_los / 100.0
		sum_los = 0
		z = vae.encoder(x, test=True)
		print z.data
