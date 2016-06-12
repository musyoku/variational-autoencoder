# -*- coding: utf-8 -*-
from args import args
from vae_m1 import BernoulliM1VAE, GaussianM1VAE, Conf

conf = Conf()
conf.gpu_enabled = True if args.gpu_enabled == 1 else False
conf.ndim_z = 50
conf.encoder_apply_dropout = False
conf.decoder_apply_dropout = False
conf.encoder_apply_batchnorm = True
conf.encoder_apply_batchnorm_to_input = True
conf.decoder_apply_batchnorm = True
conf.decoder_apply_batchnorm_to_input = True
conf.encoder_units = [600, 600]
conf.decoder_units = [600, 600]
vae = BernoulliM1VAE(conf, name="m1")
vae.load(args.model_dir)