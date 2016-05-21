# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from chainer import cuda, Variable
sys.path.append(os.path.split(os.getcwd())[0])
import util
from args import args
from model import conf, vae

dist = "bernoulli"
if isinstance(vae, GaussianM2VAE):
	dist = "gaussian"
dataset = util.load_images(args.train_image_dir, dist=dist)

max_epoch = 1000
num_trains_per_epoch = 2000
batchsize = 128
total_time = 0

for epoch in xrange(max_epoch):
	sum_loss = 0
	epoch_time = time.time()
	for t in xrange(num_trains_per_epoch):
		x = util.sample_x_variable(batchsize, conf.ndim_x, dataset, gpu_enabled=conf.gpu_enabled)
		loss = vae.train(x)
		sum_loss += loss
		if t % 100 == 0:
			sys.stdout.write("\rTraining in progress...(%d / %d)" % (t, num_trains_per_epoch))
			sys.stdout.flush()
	epoch_time = time.time() - epoch_time
	total_time += epoch_time
	sys.stdout.write("\r")
	print "epoch:", epoch, "loss:", sum_loss / num_trains_per_epoch, "time:", "%d" % (epoch_time / 60), "min", "total", "%d" % (total_time / 60), "min"
	sys.stdout.flush()
	vae.save(args.model_dir)

