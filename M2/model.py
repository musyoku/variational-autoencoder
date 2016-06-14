# -*- coding: utf-8 -*-
from args import args
from vae_m2 import BernoulliM2VAE, GaussianM2VAE, Conf

conf = Conf()
conf.gpu_enabled = True if args.gpu_enabled == 1 else False
conf.ndim_z = 50
conf.encoder_xy_z_apply_dropout = True
conf.encoder_x_y_apply_dropout = True
conf.decoder_apply_dropout = True
conf.encoder_xy_z_apply_batchnorm_to_input = False
conf.encoder_x_y_apply_batchnorm_to_input = False
conf.decoder_apply_batchnorm_to_input = False
conf.encoder_xy_z_apply_batchnorm = False
conf.encoder_x_y_apply_batchnorm = False
conf.decoder_apply_batchnorm = False
conf.encoder_xy_z_hidden_units = [500]
conf.encoder_x_y_hidden_units = [500]
conf.decoder_hidden_units = [500]
conf.batchnorm_before_activation = True if args.batchnorm_before_activation == 1 else False

if args.vae_type == "gaussian":
	vae = GaussianM2VAE(conf, name="m2")
elif args.vae_type == "bernoulli":
	vae = BernoulliM2VAE(conf, name="m2")
else:
	raise Exception()
	
vae.load(args.model_dir)