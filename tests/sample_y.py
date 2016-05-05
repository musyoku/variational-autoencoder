# -*- coding: utf-8 -*-
import sys, os, math
import numpy as np
from chainer import Variable, cuda, gradient_check
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
from vae_m2 import LabelSampler

batchsize = 2
ndim_x = 3
x = np.random.normal(size=(batchsize, ndim_x)).astype(np.float32)
x = Variable(x)
x.to_gpu()

xp = cuda.cupy
x = F.softmax(x)
print x.data

y_grad = xp.ones((batchsize, ndim_x)).astype(xp.float32)
gradient_check.check_backward(LabelSampler(), x.data, y_grad, eps=1e-2)