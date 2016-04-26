# -*- coding: utf-8 -*-
import numpy as np
import chainer
from chainer import cuda, Variable, function, FunctionSet, optimizers
from chainer import functions as F
from chainer import links as L
from activations import activations

class VAE():
	# name is used for the filename when you save the model
	def __init__(self, encoder, decoder, learning_rate=0.00025, gradient_momentum=0.95, name="vae"):
		self.encoder = encoder
		self.decoder = decoder
		self.name = name

		self.optimizer_encoder = optimizers.Adam(alpha=learning_rate, beta1=gradient_momentum)
		self.optimizer_encoder.setup(self.encoder)
		self.optimizer_encoder.add_hook(GradientClipping(10.0))

		self.optimizer_decoder = optimizers.Adam(alpha=learning_rate, beta1=gradient_momentum)
		self.optimizer_decoder.setup(self.decoder)
		self.optimizer_decoder.add_hook(GradientClipping(10.0))

	@property
	def xp(self):
		return self.encode.xp

	@property
	def gpu(self):
		return True if self.xp is cuda.cupy else False

	def encode(self, x):
		return self.encoder(x)

	def decode(self, x):
		return self.decoder(x)

	# We set L = 1
	def train(self, x, test=False):
		z_mean, z_ln_var = self.encoder(x, test=test, sample_output=False)
		# Sample z
		z = F.gaussian(z_mean, z_ln_var)
		# Decode
		x_reconstruction_mean, x_reconstruction_ln_var = self.decoder(z, test=test, sample_output=False)
		# Approximation of E_q(z|x)[log(p(x|z))]
		reconstuction_loss = F.gaussian_nll(x, x_reconstruction_mean, x_reconstruction_ln_var)
		# KL divergence
		kld_regularization_loss = F.gaussian_kl_divergence(z_mean, z_ln_var)
		loss = reconstuction_loss + kld_regularization_loss

		self.zero_grads()
		loss.backward()
		self.update()

		if self.gpu:
			loss.to_cpu()
		return loss.data

	def zero_grads(self):
		self.optimizer_encoder.zero_grads()
		self.optimizer_decoder.zero_grads()

	def update(self):
		self.optimizer_encoder.update()
		self.optimizer_decoder.update()

	def load(self, dir=None):
		if dir is None:
			raise Exception()
		for attr in vars(self):
			prop = getattr(self, attr)
			if isinstance(prop, chainer.Chain) or isinstance(prop, chainer.optimizer.GradientMethod):
				filename = dir + "/%s_%s.hdf5" % (self.name, attr)
				if os.path.isfile(filename):
					serializers.load_hdf5(filename, prop)
				else:
					print filename, "missing."
		print "model loaded."

	def save(self, dir=None):
		if dir is None:
			raise Exception()
		try:
			os.mkdir(dir)
		except:
			pass
		for attr in vars(self):
			prop = getattr(self, attr)
			if isinstance(prop, chainer.Chain) or isinstance(prop, chainer.optimizer.GradientMethod):
				serializers.save_hdf5(dir + "/%s_%s.hdf5" % (self.name, attr), prop)
		print "model saved."

class Encoder(chainer.Chain):
	def __init__(self, **layers):
		super(Encoder, self).__init__(**layers)
		self.apply_batchnorm_to_input = True
		self.apply_batchnorm = True
		self.apply_dropout = True

	@property
	def xp(self):
		return np if self.layer_0._cpu else cuda.cupy

	def forward_one_step(self, x, test=False, sample_output=True):
		activate = activations[self.activation_type]

		chain_mean = [x]
		chain_variance = [x]

		# Hidden
		for i in range(self.n_layers):
			u = chain_mean[-1]
			if i == 0:
				if self.apply_batchnorm_to_input:
					u = getattr(self, "batchnorm_mean_%d" % i)(u, test=test)
			else:
				if self.apply_batchnorm:	
					u = getattr(self, "batchnorm_mean_%d" % i)(u, test=test)
			output = activate(u)
			if self.apply_dropout:
				output = F.dropout(output, train=not test)
			chain_mean.append(output)

			u = chain_variance[-1]
			if i == 0:
				if self.apply_batchnorm_to_input:
					u = getattr(self, "batchnorm_variance_%i" % i)(u, test=test)
			else:
				if self.apply_batchnorm:
					u = getattr(self, "batchnorm_variance_%i" % i)(u, test=test)
			output = activate(u)
			if self.apply_dropout:
				output = F.dropout(output, train=not test)
			chain_variance.append(output)

		mean = chain_mean[-1]
		# log(sigma^2)
		ln_var = chain_variance[-1]

		if sample_output:
			return F.gaussian(mean, ln_var)
		return mean, ln_var

class Decder(chainer.Chain):
	def __init__(self, **layers):
		super(Decder, self).__init__(**layers)

	def forward_one_step(self, x, test=False, sample_output=True):
		activate = activations[self.activation_type]

		chain_mean = [x]
		chain_variance = [x]

		# Hidden
		for i in range(self.n_layers - 1):
			u = getattr(self, "layer_mean_%i" % i)(chain_mean[-1])
			if i == 0:
				if self.apply_batchnorm_to_input:
					u = getattr(self, "batchnorm_mean_%i" % i)(u, test=test)
			else:
				if self.apply_batchnorm:
					u = getattr(self, "batchnorm_mean_%i" % i)(u, test=test)
			output = activate(u)
			if self.apply_dropout:
				output = F.dropout(output, train=not test)
			chain_mean.append(output)

			u = getattr(self, "layer_variance_%i" % i)(chain_variance[-1])
			if i == 0:
				if self.apply_batchnorm_to_input:
					u = getattr(self, "batchnorm_variance_%i" % i)(u, test=test)
			else:
				if self.apply_batchnorm:
					u = getattr(self, "batchnorm_variance_%i" % i)(u, test=test)
			output = activate(u)
			if self.apply_dropout:
				output = F.dropout(output, train=not test)
			chain_variance.append(output)

		# Output
		u = getattr(self, "layer_mean_%i" % (self.n_layers - 1))(chain_mean[-1])
		if self.apply_batchnorm_to_output:
			u = getattr(self, "batchnorm_mean_%i" % (self.n_layers - 1))(u, test=test)
		if self.output_activation_type is None:
			chain_mean.append(u)
		else:
			chain_mean.append(activations[self.output_activation_type](u))

		u = getattr(self, "layer_variance_%i" % (self.n_layers - 1))(chain_variance[-1])
		if self.apply_batchnorm_to_output:
			u = getattr(self, "batchnorm_variance_%i" % (self.n_layers - 1))(u, test=test)
		if self.output_activation_type is None:
			chain_variance.append(u)
		else:
			chain_variance.append(activations[self.output_activation_type](u))

		mean = chain_mean[-1]
		# log(sigma^2)
		ln_var = chain_variance[-1]

		if sample_output:
			return F.gaussian(mean, ln_var)
		return mean, ln_var
