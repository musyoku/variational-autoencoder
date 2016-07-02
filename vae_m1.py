# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six
from chainer import cuda, Variable, optimizers, serializers, optimizer
from chainer import functions as F
from chainer import links as L

activations = {
	"sigmoid": F.sigmoid, 
	"tanh": F.tanh, 
	"softplus": F.softplus, 
	"relu": F.relu, 
	"leaky_relu": F.leaky_relu, 
	"elu": F.elu
}

class Conf():
	def __init__(self):
		self.image_width = 28
		self.image_height = 28
		self.ndim_x = 28 * 28
		self.ndim_z = 100
		self.batchnorm_before_activation = True

		# gaussianmarg | gaussian
		# We recommend you to use "gaussianmarg" when decoder is gaussian.
		self.type_pz = "gaussianmarg"
		self.type_qz = "gaussianmarg"

		# e.g.
		# ndim_x (input) -> 2000 -> 1000 -> 100 (output)
		# encoder_hidden_units = [2000, 1000]
		self.encoder_hidden_units = [600, 600]
		self.encoder_activation_function = "softplus"
		self.encoder_apply_dropout = True
		self.encoder_apply_batchnorm = True
		self.encoder_apply_batchnorm_to_input = True

		self.decoder_hidden_units = [600, 600]
		self.decoder_activation_function = "softplus"
		self.decoder_apply_dropout = True
		self.decoder_apply_batchnorm = True
		self.decoder_apply_batchnorm_to_input = True

		self.gpu_enabled = True
		self.learning_rate = 0.0003
		self.gradient_momentum = 0.9
		self.gradient_clipping = 1.0

	def check(self):
		pass

def sum_sqnorm(arr):
	sq_sum = collections.defaultdict(float)
	for x in arr:
		with cuda.get_device(x) as dev:
			x = x.ravel()
			s = x.dot(x)
			sq_sum[int(dev)] += s
	return sum([float(i) for i in six.itervalues(sq_sum)])
	
class GradientClipping(object):
	name = "GradientClipping"

	def __init__(self, threshold):
		self.threshold = threshold

	def __call__(self, opt):
		norm = np.sqrt(sum_sqnorm([p.grad for p in opt.target.params()]))
		if norm < 1:
			return
		rate = self.threshold / norm
		if rate < 1:
			for param in opt.target.params():
				grad = param.grad
				with cuda.get_device(grad):
					grad = cuda.cupy.clip(grad, -self.threshold, self.threshold)

class VAE():
	# name is used for the filename when you save the model
	def __init__(self, conf, name="vae"):
		conf.check()
		self.encoder, self.decoder = self.build(conf)
		self.name = name

		self.optimizer_encoder = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_encoder.setup(self.encoder)
		# self.optimizer_encoder.add_hook(optimizer.WeightDecay(0.001))
		self.optimizer_encoder.add_hook(GradientClipping(conf.gradient_clipping))

		self.optimizer_decoder = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_decoder.setup(self.decoder)
		# self.optimizer_decoder.add_hook(optimizer.WeightDecay(0.001))
		self.optimizer_decoder.add_hook(GradientClipping(conf.gradient_clipping))

		self.type_pz = conf.type_pz
		self.type_qz = conf.type_qz
		
	def build(self, conf):
		raise Exception()

	def train(self, x, L=1, test=False):
		raise Exception()

	@property
	def xp(self):
		return self.encoder.xp

	@property
	def gpu(self):
		if cuda.available is False:
			return False
		return True if self.xp is cuda.cupy else False

	def zero_grads(self):
		self.optimizer_encoder.zero_grads()
		self.optimizer_decoder.zero_grads()

	def update(self):
		self.optimizer_encoder.update()
		self.optimizer_decoder.update()

	def bernoulli_nll_keepbatch(self, x, y):
		nll = F.softplus(y) - x * y
		return F.sum(nll, axis=1)

	def gaussian_nll_keepbatch(self, x, mean, ln_var):
		x_prec = F.exp(-ln_var)
		x_diff = x - mean
		x_power = x_diff ** 2 * x_prec * 0.5
		return F.sum((math.log(2.0 * math.pi) + ln_var) * 0.5 + x_power, axis=1)

	def gaussian_kl_divergence_keepbatch(self, mean, ln_var):
		var = F.exp(ln_var)
		kld = F.sum(mean ** 2 + var - ln_var - 1, axis=1) * 0.5
		return kld

	def log_px_z(self, x, z, test=False):
		if isinstance(self.decoder, BernoulliDecoder):
			# do not apply F.sigmoid to the output of the decoder
			raw_output = self.decoder(z, test=test, apply_f=False)
			negative_log_likelihood = self.bernoulli_nll_keepbatch(x, raw_output)
			log_px_z = -negative_log_likelihood
		else:
			x_mean, x_ln_var = self.decoder(z, test=test, apply_f=False)
			negative_log_likelihood = self.gaussian_nll_keepbatch(x, x_mean, x_ln_var)
			log_px_z = -negative_log_likelihood
		return log_px_z

	# this will not be used for bernoulli decoder
	def log_pz(self, z, mean, ln_var):
		if self.type_pz == "gaussianmarg":
			# \int q(z)logp(z)dz = -(J/2)*log2pi - (1/2)*sum_{j=1}^{J} (mu^2 + var)
			# See Appendix B [Auto-Encoding Variational Bayes](http://arxiv.org/abs/1312.6114)
			# See https://github.com/dpkingma/nips14-ssl/blob/master/anglepy/models/VAE_YZ_X.py line 106
			log_pz = -0.5 * (math.log(2.0 * math.pi) + mean * mean + F.exp(ln_var))
		elif self.type_pz == "gaussian":
			log_pz = -0.5 * math.log(2.0 * math.pi) - 0.5 * z ** 2
		return F.sum(log_pz, axis=1)

	# this will not be used for bernoulli decoder
	def log_qz_x(self, z, mean, ln_var):
		if self.type_qz == "gaussianmarg":
			# \int q(z)logq(z)dz = -(J/2)*log2pi - (1/2)*sum_{j=1}^{J} (1 + logvar)
			# See Appendix B [Auto-Encoding Variational Bayes](http://arxiv.org/abs/1312.6114)
			# See https://github.com/dpkingma/nips14-ssl/blob/master/anglepy/models/VAE_YZ_X.py line 118
			log_qz_x = -0.5 * F.sum((math.log(2.0 * math.pi) + 1 + ln_var), axis=1)
		elif self.type_qz == "gaussian":
			log_qz_x = -self.gaussian_nll_keepbatch(z, mean, ln_var)
		return log_qz_x

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

class GaussianM1VAE(VAE):

	def build(self, conf):
		wscale = 0.1
		encoder_attributes = {}
		encoder_units = [(conf.ndim_x, conf.encoder_hidden_units[0])]
		encoder_units += zip(conf.encoder_hidden_units[:-1], conf.encoder_hidden_units[1:])
		for i, (n_in, n_out) in enumerate(encoder_units):
			encoder_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			if conf.batchnorm_before_activation:
				encoder_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				encoder_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)
		encoder_attributes["layer_mean"] = L.Linear(conf.encoder_hidden_units[-1], conf.ndim_z, wscale=wscale)
		encoder_attributes["layer_var"] = L.Linear(conf.encoder_hidden_units[-1], conf.ndim_z, wscale=wscale)
		encoder = Encoder(**encoder_attributes)
		encoder.n_layers = len(encoder_units)
		encoder.activation_function = conf.encoder_activation_function
		encoder.apply_dropout = conf.encoder_apply_dropout
		encoder.apply_batchnorm = conf.encoder_apply_batchnorm
		encoder.apply_batchnorm_to_input = conf.encoder_apply_batchnorm_to_input
		encoder.batchnorm_before_activation = conf.batchnorm_before_activation

		decoder_attributes = {}
		decoder_units = [(conf.ndim_z, conf.decoder_hidden_units[0])]
		decoder_units += zip(conf.decoder_hidden_units[:-1], conf.decoder_hidden_units[1:])
		for i, (n_in, n_out) in enumerate(decoder_units):
			decoder_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			if conf.batchnorm_before_activation:
				decoder_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				decoder_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)
		decoder_attributes["layer_mean"] = L.Linear(conf.decoder_hidden_units[-1], conf.ndim_x, wscale=wscale)
		decoder_attributes["layer_var"] = L.Linear(conf.decoder_hidden_units[-1], conf.ndim_x, wscale=wscale)
		decoder = GaussianDecoder(**decoder_attributes)
		decoder.n_layers = len(decoder_units)
		decoder.activation_function = conf.decoder_activation_function
		decoder.apply_dropout = conf.decoder_apply_dropout
		decoder.apply_batchnorm = conf.decoder_apply_batchnorm
		decoder.apply_batchnorm_to_input = conf.decoder_apply_batchnorm_to_input
		decoder.batchnorm_before_activation = conf.batchnorm_before_activation

		if conf.gpu_enabled:
			encoder.to_gpu()
			decoder.to_gpu()
		return encoder, decoder

	def train(self, x, L=1, test=False):
		batchsize = x.data.shape[0]
		z_mean, z_ln_var = self.encoder(x, test=test, apply_f=False)
		loss = 0
		for l in xrange(L):
			# Sample z
			z = F.gaussian(z_mean, z_ln_var)

			# Compute lower bound
			log_px_z = self.log_px_z(x, z, test=test)
			log_pz = self.log_pz(z, z_mean, z_ln_var)
			log_qz_x = self.log_qz_x(z, z_mean, z_ln_var)
			lower_bound = log_px_z + log_pz - log_qz_x

			loss += -lower_bound

		loss = F.sum(loss) / L / batchsize

		self.zero_grads()
		loss.backward()
		self.update()

		if self.gpu:
			loss.to_cpu()
		return loss.data

class BernoulliM1VAE(VAE):

	def build(self, conf):
		wscale = 0.1
		encoder_attributes = {}
		encoder_units = [(conf.ndim_x, conf.encoder_hidden_units[0])]
		encoder_units += zip(conf.encoder_hidden_units[:-1], conf.encoder_hidden_units[1:])
		for i, (n_in, n_out) in enumerate(encoder_units):
			encoder_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			if conf.batchnorm_before_activation:
				encoder_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				encoder_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)
		encoder_attributes["layer_mean"] = L.Linear(conf.encoder_hidden_units[-1], conf.ndim_z, wscale=wscale)
		encoder_attributes["layer_var"] = L.Linear(conf.encoder_hidden_units[-1], conf.ndim_z, wscale=wscale)
		encoder = Encoder(**encoder_attributes)
		encoder.n_layers = len(encoder_units)
		encoder.activation_function = conf.encoder_activation_function
		encoder.apply_dropout = conf.encoder_apply_dropout
		encoder.apply_batchnorm = conf.encoder_apply_batchnorm
		encoder.apply_batchnorm_to_input = conf.encoder_apply_batchnorm_to_input
		encoder.batchnorm_before_activation = conf.batchnorm_before_activation

		decoder_attributes = {}
		decoder_units = [(conf.ndim_z, conf.decoder_hidden_units[0])]
		decoder_units += zip(conf.decoder_hidden_units[:-1], conf.decoder_hidden_units[1:])
		decoder_units += [(conf.decoder_hidden_units[-1], conf.ndim_x)]
		for i, (n_in, n_out) in enumerate(decoder_units):
			decoder_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			if conf.batchnorm_before_activation:
				decoder_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				decoder_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)
		decoder = BernoulliDecoder(**decoder_attributes)
		decoder.n_layers = len(decoder_units)
		decoder.activation_function = conf.decoder_activation_function
		decoder.apply_dropout = conf.decoder_apply_dropout
		decoder.apply_batchnorm = conf.decoder_apply_batchnorm
		decoder.apply_batchnorm_to_input = conf.decoder_apply_batchnorm_to_input
		decoder.batchnorm_before_activation = conf.batchnorm_before_activation

		if conf.gpu_enabled:
			encoder.to_gpu()
			decoder.to_gpu()
		return encoder, decoder

	def train(self, x, L=1, test=False):
		batchsize = x.data.shape[0]
		z_mean, z_ln_var = self.encoder(x, test=test, apply_f=False)
		loss = 0
		for l in xrange(L):
			# Sample z
			z = F.gaussian(z_mean, z_ln_var)
			# Decode
			x_expectation = self.decoder(z, test=test, apply_f=False)
			# E_q(z|x)[log(p(x|z))]
			loss += self.bernoulli_nll_keepbatch(x, x_expectation)
		if L > 1:
			loss /= L
		# KL divergence
		loss += self.gaussian_kl_divergence_keepbatch(z_mean, z_ln_var)
		loss = F.sum(loss) / batchsize

		self.zero_grads()
		loss.backward()
		self.update()

		if self.gpu:
			loss.to_cpu()
		return loss.data

class Encoder(chainer.Chain):
	def __init__(self, **layers):
		super(Encoder, self).__init__(**layers)
		self.activation_function = "softplus"
		self.apply_batchnorm_to_input = True
		self.apply_batchnorm = True
		self.apply_dropout = True
		self.batchnorm_before_activation = True

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def forward_one_step(self, x, test=False, apply_f=True):
		f = activations[self.activation_function]

		chain = [x]

		# Hidden
		for i in range(self.n_layers):
			u = chain[-1]
			if self.batchnorm_before_activation:
				u = getattr(self, "layer_%i" % i)(u)
			if i == 0:
				if self.apply_batchnorm_to_input:
					u = getattr(self, "batchnorm_%d" % i)(u, test=test)
			else:
				if self.apply_batchnorm:
					u = getattr(self, "batchnorm_%d" % i)(u, test=test)
			if self.batchnorm_before_activation == False:
				u = getattr(self, "layer_%i" % i)(u)
			output = f(u)
			if self.apply_dropout:
				output = F.dropout(output, train=not test)
			chain.append(output)

		u = chain[-1]
		mean = self.layer_mean(u)

		# log(sigma^2)
		u = chain[-1]
		ln_var = self.layer_var(u)

		return mean, ln_var

	def __call__(self, x, test=False, apply_f=True):
		mean, ln_var = self.forward_one_step(x, test=test, apply_f=apply_f)
		if apply_f:
			return F.gaussian(mean, ln_var)
		return mean, ln_var

# Network structure is same as the Encoder
class GaussianDecoder(Encoder):

	def __call__(self, x, test=False, apply_f=False):
		mean, ln_var = self.forward_one_step(x, test=test, apply_f=False)
		if apply_f:
			return F.gaussian(mean, ln_var)
		return mean, ln_var

class BernoulliDecoder(chainer.Chain):
	def __init__(self, **layers):
		super(BernoulliDecoder, self).__init__(**layers)
		self.activation_function = "softplus"
		self.apply_batchnorm_to_input = True
		self.apply_batchnorm = True
		self.apply_dropout = True
		self.batchnorm_before_activation = True

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def forward_one_step(self, x, test=False):
		f = activations[self.activation_function]
		chain = [x]

		# Hidden
		for i in range(self.n_layers):
			u = chain[-1]
			if self.batchnorm_before_activation:
				u = getattr(self, "layer_%i" % i)(u)
			if i == 0:
				if self.apply_batchnorm_to_input:
					u = getattr(self, "batchnorm_%d" % i)(u, test=test)
			elif i == self.n_layers - 1:
				if self.apply_batchnorm_to_input and self.batchnorm_before_activation == False:
					u = getattr(self, "batchnorm_%d" % i)(u, test=test)
			else:
				if self.apply_batchnorm:
					u = getattr(self, "batchnorm_%d" % i)(u, test=test)
			if self.batchnorm_before_activation == False:
				u = getattr(self, "layer_%i" % i)(u)
			if i == self.n_layers - 1:
				output = u
			else:
				output = f(u)
				if self.apply_dropout:
					output = F.dropout(output, train=not test)
			chain.append(output)

		return chain[-1]

	def __call__(self, x, test=False, apply_f=False):
		output = self.forward_one_step(x, test=test)
		if apply_f:
			return F.sigmoid(output)
		return output