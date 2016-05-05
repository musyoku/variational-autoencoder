# -*- coding: utf-8 -*-
from args import args
from vae_m1 import BernoulliM1VAE, GaussianM1VAE, Conf

conf = Conf()
conf.use_gpu = False if args.use_gpu == -1 else True
conf.ndim_z = 50
conf.encoder_apply_batchnorm_to_input = True
conf.encoder_apply_batchnorm = False
conf.decoder_apply_batchnorm_to_input = True
conf.decoder_apply_batchnorm = False
conf.encoder_units = [600, 600]
conf.decoder_units = [600, 600]
vae = BernoulliM1VAE(conf, name="m1")