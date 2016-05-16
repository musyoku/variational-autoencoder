# -*- coding: utf-8 -*-
import sys, os, math
import numpy as np
from chainer import Variable
from chainer import functions as F

batchsize = 10
ndim_x = 10

# gaussian
print "gaussian:"
x_batch = np.random.uniform(-10, 10, (batchsize, ndim_x)).astype(np.float32)
x_batch = Variable(x_batch)
mean = np.random.uniform(-1, 1, (batchsize, ndim_x)).astype(np.float32)
ln_var = np.random.uniform(-10, 10, (batchsize, ndim_x)).astype(np.float32)
mean = Variable(mean)
ln_var = Variable(ln_var)
nll = F.gaussian_nll(x_batch, mean, ln_var)
print nll.data
nll = F.exp(-nll)
print nll.data

# a = 1 / math.sqrt(2 * math.pi * variance) * math.exp(-(x - mu) ** 2 / (2 * variance))
# print -math.log(a + 1e-6)
# print math.exp(math.log(a + 1e-6))

# kl divergence
print "kl divergence"

J = mean.data.size
print J
var = F.exp(ln_var)
kld = (F.sum(mean * mean) + F.sum(var) - F.sum(ln_var) - J) * 0.5
print kld.data

kld = F.sum(mean * mean + var - ln_var - mean.data.shape[0], axis=1) * 0.5
print kld.data



# bernoulli
print "bernoulli:"
x_batch = np.random.uniform(0, 1, (batchsize, ndim_x)).astype(np.float32)
x_batch = Variable(x_batch)
y_batch = np.random.uniform(-10, 10, (batchsize, ndim_x)).astype(np.float32)
y_batch = Variable(y_batch)
nll = F.bernoulli_nll(x_batch, y_batch)
print nll.data

p_batch = F.sigmoid(y_batch)
nll = -(x_batch * F.log(p_batch) + (1 - x_batch) * F.log(1 - p_batch))
print np.sum(nll.data)
nll = F.softplus(y_batch) - x_batch * y_batch
print F.sum(nll, axis=1).data
print np.sum(nll.data)