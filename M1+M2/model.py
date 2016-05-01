# -*- coding: utf-8 -*-
from args import args
from vae import BernoulliVAE, GaussianVAE, Conf

conf = Conf()
conf.use_gpu = False if args.use_gpu == -1 else True
conf.ndim_z = 50
conf.encoder_apply_batchnorm_to_input = True
conf.decoder_apply_batchnorm_to_input = True
vae = BernoulliVAE(conf, name="m1")