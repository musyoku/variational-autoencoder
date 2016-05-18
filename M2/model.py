# -*- coding: utf-8 -*-
from args import args
from vae_m2 import BernoulliM2VAE, GaussianM2VAE, Conf

conf = Conf()
conf.use_gpu = False if args.use_gpu == 0 else True
conf.ndim_z = 50
conf.encoder_xy_z_apply_batchnorm_to_input = False
conf.encoder_xy_z_apply_batchnorm = False
conf.encoder_x_y_apply_batchnorm_to_input = False
conf.encoder_x_y_apply_batchnorm = False
conf.decoder_apply_batchnorm_to_input = False
conf.decoder_apply_batchnorm = False
conf.encoder_xy_z_hidden_units = [500]
conf.encoder_x_y_hidden_units = [500]
conf.decoder_hidden_units = [500]
vae = BernoulliM2VAE(conf, name="m2")