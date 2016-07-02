# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from chainer import cuda, Variable
sys.path.append(os.path.split(os.getcwd())[0])
import util
from args import args
from model import conf, vae
from vae_m1 import GaussianM1VAE

dist = "bernoulli"
if isinstance(vae, GaussianM1VAE):
	dist = "gaussian"
dataset = util.load_images(args.train_image_dir, dist=dist)

max_epoch = 1000
num_trains_per_epoch = 2000
batchsize = 100
total_time = 0

for epoch in xrange(max_epoch):
	sum_loss = 0
	epoch_time = time.time()
	for t in xrange(num_trains_per_epoch):
		x = util.sample_x_variable(batchsize, conf.ndim_x, dataset, gpu_enabled=conf.gpu_enabled)

		# train
		loss = vae.train(x, L=1)

		sum_loss += loss
		if t % 10 == 0:
			sys.stdout.write("\rTraining M1 in progress...(%d / %d)" % (t, num_trains_per_epoch))
			sys.stdout.flush()
	epoch_time = time.time() - epoch_time
	total_time += epoch_time
	sys.stdout.write("\r")
	print "epoch:", epoch, "loss: {:.3f}".format(sum_loss / num_trains_per_epoch), "time: {:d} min".format(int(epoch_time / 60)), "total: {:d} min".format(int(total_time / 60))
	sys.stdout.flush()
	vae.save(args.model_dir)

