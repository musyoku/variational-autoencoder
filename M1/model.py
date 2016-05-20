# -*- coding: utf-8 -*-
from args import args
from vae_m1 import BernoulliM1VAE, GaussianM1VAE, Conf

conf = Conf()
conf.gpu_enabled = True if args.gpu_enabled == 1 else False
conf.ndim_z = 50
conf.encoder_apply_dropout = True
conf.decoder_apply_dropout = True
conf.encoder_units = [600, 600]
conf.decoder_units = [600, 600]
vae = BernoulliM1VAE(conf, name="m1")