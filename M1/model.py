# -*- coding: utf-8 -*-
from args import args
from vae import BernoulliVAE, GaussianVAE, Conf

conf = Conf()
conf.use_gpu = False if args.use_gpu == -1 else True
conf.ndim_z = 10
conf.encoder_units = [conf.ndim_x, 100, 50, conf.ndim_z]
conf.decoder_units = [conf.ndim_z, 100, 50, conf.ndim_x]
vae = GaussianVAE(conf, name="m1")