# -*- coding: utf-8 -*-
from args import args
from vae import VAE, Conf

conf = Conf()
conf.use_gpu = args.use_gpu
vae = VAE(conf, name="M1")