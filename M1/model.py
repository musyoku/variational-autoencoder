# -*- coding: utf-8 -*-
from args import args
from vae import BernoulliVAE, Conf

conf = Conf()
conf.use_gpu = args.use_gpu
vae = BernoulliVAE(conf, name="m1")