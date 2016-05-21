# -*- coding: utf-8 -*-
import sys, os, math
import numpy as np
from chainer import Variable
from chainer import functions as F
from sklearn import preprocessing

batchsize = 10
ndim_x = 10

# gaussian

def gaussian_nll_keepbatch(x, mean, ln_var):
	x_prec = F.exp(-ln_var)
	x_diff = x - mean
	x_power = (x_diff * x_diff) * x_prec * 0.5
	return F.sum((math.log(2.0 * math.pi) + ln_var) * 0.5 + x_power, axis=1)

print "gaussian:"
x_batch = np.random.uniform(-10, 10, (batchsize, ndim_x)).astype(np.float32)
x_batch = Variable(x_batch)
mean_batch = np.random.uniform(-1, 1, (batchsize, ndim_x)).astype(np.float32)
ln_var_batch = np.random.uniform(-10, 10, (batchsize, ndim_x)).astype(np.float32)
mean_batch = Variable(mean_batch)
ln_var_batch = Variable(ln_var_batch)
print "chainer::"
for i in xrange(batchsize):
	x = Variable(x_batch.data[i].reshape(1, -1))
	mean = Variable(mean_batch.data[i].reshape(1, -1))
	ln_var = Variable(ln_var_batch.data[i].reshape(1, -1))
	nll = F.gaussian_nll(x, mean, ln_var)
	print nll.data

print "keepbatch::"
nll = gaussian_nll_keepbatch(x_batch, mean_batch, ln_var_batch)
print nll.data

# kl divergence
def gaussian_kl_divergence(mean, ln_var):
	var = F.exp(ln_var)
	kld = F.sum(mean * mean + var - ln_var - 1, axis=1) * 0.5
	return kld

print "kl divergence"
print "chainer::"
for i in xrange(batchsize):
	mean = Variable(mean_batch.data[i].reshape(1, -1))
	ln_var = Variable(ln_var_batch.data[i].reshape(1, -1))
	kld = F.gaussian_kl_divergence(mean, ln_var)
	print kld.data

print "keepbatch::"
kld = gaussian_kl_divergence(mean_batch, ln_var_batch)
print kld.data

# bernoulli
def bernoulli_nll_keepbatch(x, y):
	nll = F.softplus(y) - x * y
	return F.sum(nll, axis=1)

print "bernoulli:"
x_batch = preprocessing.binarize(np.random.uniform(0, 1, (batchsize, ndim_x)).astype(np.float32), threshold=0.5)
x_batch = Variable(x_batch)
y_batch = np.random.uniform(0, 1, (batchsize, ndim_x)).astype(np.float32)
y_batch = Variable(y_batch)
print "chainer::"
for i in xrange(batchsize):
	x = Variable(x_batch.data[i].reshape(1, -1))
	y = Variable(y_batch.data[i].reshape(1, -1))
	nll = F.bernoulli_nll(x, y)
	print nll.data

print "keepbatch::"
nll = bernoulli_nll_keepbatch(x_batch, y_batch)
print nll.data