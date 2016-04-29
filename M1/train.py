# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from chainer import cuda, Variable
sys.path.append(os.path.split(os.getcwd())[0])
import util
from args import args
from model import conf, vae

vae.load(args.model_dir)
dataset, labels = util.load_labeled_images(args)

max_epoch = 1000
num_trains_per_epoch = 500
batchsize = 128
total_time = 0

for epoch in xrange(max_epoch):
	sum_loss = 0
	epoch_time = time.time()
	for t in xrange(num_trains_per_epoch):
		x = util.sample_x_variables(batchsize, conf.ndim_x, dataset, use_gpu=conf.use_gpu)
		loss = vae.train(x)
		print loss
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

