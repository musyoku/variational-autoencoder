# -*- coding: utf-8 -*-
import sys, os, math
import numpy as np
from chainer import Variable
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
import util
from args import args
from vae import BernoulliVAE, GaussianVAE, Conf

dataset, labels = util.load_labeled_images(args)

conf = Conf()
conf.use_gpu = False
batchsize = 1
conf.ndim_x = 1
conf.ndim_z = 10
conf.encoder_units = [conf.ndim_x, 100, 50, conf.ndim_z]
conf.decoder_units = [conf.ndim_z, 100, 50, conf.ndim_x]

vae = GaussianVAE(conf, name="m1")
# batch = util.sample_x_variables(batchsize, conf.ndim_x, dataset, use_gpu=conf.use_gpu)
# mean = np.zeros((batchsize, conf.ndim_x), dtype=np.float32)
# ln_var = np.random.normal(size=(batchsize, conf.ndim_x)).astype(np.float32)
# mean = Variable(mean)
# ln_var = Variable(ln_var)
# nll = F.gaussian_nll(batch, mean, ln_var)
# print nll.data / batchsize

# batch = util.sample_x_variables(batchsize, conf.ndim_x, dataset, use_gpu=conf.use_gpu)
x = -1.0
batch = np.full((batchsize, conf.ndim_x), x, dtype=np.float32)
batch = Variable(batch)
mu = -1.0
mean = np.full((batchsize, conf.ndim_x), mu, dtype=np.float32)
variance = 0.004
ln_var = np.full((batchsize, conf.ndim_x), math.log(variance), dtype=np.float32)
mean = Variable(mean)
ln_var = Variable(ln_var)
nll = F.gaussian_nll(batch, mean, ln_var)
print nll.data
nll = F.exp(-nll)
print nll.data

a = 1 / math.sqrt(2 * math.pi * variance) * math.exp(-(x - mu) ** 2 / (2 * variance))
print -math.log(a + 1e-6)
print math.exp(math.log(a + 1e-6))

# for t in xrange(1000):
# 	loss = vae.train(batch)
# 	print "nll", loss, "p", math.exp(-loss)