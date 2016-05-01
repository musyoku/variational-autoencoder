# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six
from chainer import cuda, Variable, optimizers, serializers
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
		self.ndim_y = 10
		self.ndim_z = 100

		# ie.
		# 784+10(input vector) -> 2000 -> 1000 -> 100(output vector)
		# encoder_xy_z_units = [794, 2000, 1000, 100]
		self.encoder_xy_z_units = [self.ndim_x + self.ndim_y, 512, 256, self.ndim_z]
		self.encoder_xy_z_activation_function = "softplus"
		self.encoder_xy_z_output_activation_function = None
		self.encoder_xy_z_apply_dropout = False
		self.encoder_xy_z_apply_batchnorm = False
		self.encoder_xy_z_apply_batchnorm_to_input = False

		self.encoder_x_y_units = [self.ndim_x, 512, 256, self.ndim_y]
		self.encoder_x_y_activation_function = "softplus"
		self.encoder_x_y_output_activation_function = None
		self.encoder_x_y_apply_dropout = False
		self.encoder_x_y_apply_batchnorm = False
		self.encoder_x_y_apply_batchnorm_to_input = False

		self.decoder_units = [self.ndim_z + self.ndim_y, 256, 512, self.ndim_x]
		self.decoder_activation_function = "softplus"
		self.decoder_output_activation_function = None	# this will be ignored when decoder is BernoulliDecoder
		self.decoder_apply_dropout = False
		self.decoder_apply_batchnorm = False
		self.decoder_apply_batchnorm_to_input = False

		self.use_gpu = True
		self.learning_rate = 0.00025
		self.gradient_momentum = 0.95

	def check(self):
		if self.ndim_x != self.encoder_x_y_units[0]:
			raise Exception("ndim_x != encoder_x_y_units[0]")

		if self.ndim_y != self.encoder_x_y_units[-1]:
			raise Exception("ndim_x != encoder_x_y_units[-1]")

		if self.ndim_y + self.ndim_x != self.encoder_xy_z_units[0]:
			raise Exception("ndim_y + ndim_x != encoder_xy_z_units[0]")

		if self.ndim_z != self.encoder_xy_z_units[-1]:
			raise Exception("ndim_x != encoder_xy_z_units[-1]")

		if self.ndim_y + self.ndim_z != self.decoder_units[0]:
			raise Exception("ndim_y + ndim_z != decoder_units[0]")

		if self.ndim_x != self.decoder_units[-1]:
			raise Exception("ndim_x != decoder_units[-1]")

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
		if norm == 0:
			norm = 1
		rate = self.threshold / norm
		if rate < 1:
			for param in opt.target.params():
				grad = param.grad
				with cuda.get_device(grad):
					grad *= rate

class VAE():
	# name is used for the filename when you save the model
	def __init__(self, conf, name="vae"):
		conf.check()
		self.encoder_xy_z, self.encoder_x_y, self.decoder = self.build(conf)
		self.name = name

		self.optimizer_encoder_xy_z = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_encoder_xy_z.setup(self.encoder_xy_z)
		self.optimizer_encoder_xy_z.add_hook(GradientClipping(10.0))

		self.optimizer_encoder_x_y = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_encoder_x_y.setup(self.encoder_x_y)
		self.optimizer_encoder_x_y.add_hook(GradientClipping(10.0))

		self.optimizer_decoder = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_decoder.setup(self.decoder)
		self.optimizer_decoder.add_hook(GradientClipping(10.0))

	def build(self, conf):
		raise Exception()

	def train(self, x, L=1, test=False):
		raise Exception()

	@property
	def xp(self):
		return self.encoder_xy_z.xp

	@property
	def gpu(self):
		if cuda.available is False:
			return False
		return True if self.xp is cuda.cupy else False

	def zero_grads(self):
		self.optimizer_encoder_xy_z.zero_grads()
		self.optimizer_encoder_x_y.zero_grads()
		self.optimizer_decoder.zero_grads()

	def update(self):
		self.optimizer_encoder_xy_z.update()
		self.optimizer_encoder_x_y.update()
		self.optimizer_decoder.update()

	def encode(self, x, test=False):
		return self.encoder(x, test=test)

	def decode(self, z, test=False, output_pixel_value=True):
		return self.decoder(z, test=test, output_pixel_value=output_pixel_value)

	def __call__(self, x, test=False, output_pixel_value=True):
		return self.decoder(self.encoder(x, test=test), test=test, output_pixel_value=True)

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

class GaussianM2VAE(VAE):

	def build(self, conf):
		wscale = 0.1
		encoder_xy_z_attributes = {}
		encoder_xy_z_units = zip(conf.encoder_xy_z_units[:-1], conf.encoder_xy_z_units[1:])
		for i, (n_in, n_out) in enumerate(encoder_xy_z_units):
			encoder_xy_z_attributes["layer_mean_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			encoder_xy_z_attributes["batchnorm_mean_%i" % i] = L.BatchNormalization(n_in)
			encoder_xy_z_attributes["layer_var_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			encoder_xy_z_attributes["batchnorm_var_%i" % i] = L.BatchNormalization(n_in)
		encoder = GaussianEncoder(**encoder_xy_z_attributes)
		encoder.n_layers = len(encoder_xy_z_units)
		encoder.activation_function = conf.encoder_xy_z_activation_function
		encoder.output_activation_function = conf.encoder_xy_z_output_activation_function
		encoder.apply_dropout = conf.encoder_xy_z_apply_dropout
		encoder.apply_batchnorm = conf.encoder_xy_z_apply_batchnorm
		encoder.apply_batchnorm_to_input = conf.encoder_xy_z_apply_batchnorm_to_input

		decoder_attributes = {}
		decoder_units = zip(conf.decoder_units[:-1], conf.decoder_units[1:])
		for i, (n_in, n_out) in enumerate(decoder_units):
			decoder_attributes["layer_mean_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			decoder_attributes["batchnorm_mean_%i" % i] = L.BatchNormalization(n_in)
			decoder_attributes["layer_var_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			decoder_attributes["batchnorm_var_%i" % i] = L.BatchNormalization(n_in)
		decoder = GaussianDecoder(**decoder_attributes)
		decoder.n_layers = len(decoder_units)
		decoder.activation_function = conf.decoder_activation_function
		decoder.output_activation_function = conf.decoder_output_activation_function
		decoder.apply_dropout = conf.decoder_apply_dropout
		decoder.apply_batchnorm = conf.decoder_apply_batchnorm
		decoder.apply_batchnorm_to_input = conf.decoder_apply_batchnorm_to_input

		if conf.use_gpu:
			encoder.to_gpu()
			decoder.to_gpu()
		return encoder, decoder

	def train(self, x, L=1, test=False):
		z_mean, z_ln_var = self.encoder(x, test=test, sample_output=False)
		reconstuction_loss = 0
		for l in xrange(L):
			# Sample z
			z = F.gaussian(z_mean, z_ln_var)
			# Decode
			x_reconstruction_mean, x_reconstruction_ln_var = self.decoder(z, test=test, output_pixel_value=False)
			# Approximation of E_q(z|x)[log(p(x|z))]
			reconstuction_loss += F.gaussian_nll(x, x_reconstruction_mean, x_reconstruction_ln_var)
		loss = reconstuction_loss / (L * x.data.shape[0])
		# KL divergence
		kld_regularization_loss = F.gaussian_kl_divergence(z_mean, z_ln_var)
		loss += kld_regularization_loss / x.data.shape[0]

		self.zero_grads()
		loss.backward()
		self.update()

		if self.gpu:
			loss.to_cpu()
		return loss.data

class BernoulliM2VAE(VAE):

	def build(self, conf):
		wscale = 1.0
		encoder_xy_z_attributes = {}
		encoder_xy_z_units = zip(conf.encoder_xy_z_units[:-1], conf.encoder_xy_z_units[1:])
		for i, (n_in, n_out) in enumerate(encoder_xy_z_units):
			encoder_xy_z_attributes["layer_mean_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			encoder_xy_z_attributes["batchnorm_mean_%i" % i] = L.BatchNormalization(n_in)
			encoder_xy_z_attributes["layer_var_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			encoder_xy_z_attributes["batchnorm_var_%i" % i] = L.BatchNormalization(n_in)
		encoder_xy_z = GaussianEncoder(**encoder_xy_z_attributes)
		encoder_xy_z.n_layers = len(encoder_xy_z_units)
		encoder_xy_z.activation_function = conf.encoder_xy_z_activation_function
		encoder_xy_z.output_activation_function = conf.encoder_xy_z_output_activation_function
		encoder_xy_z.apply_dropout = conf.encoder_xy_z_apply_dropout
		encoder_xy_z.apply_batchnorm = conf.encoder_xy_z_apply_batchnorm
		encoder_xy_z.apply_batchnorm_to_input = conf.encoder_xy_z_apply_batchnorm_to_input

		encoder_x_y_attributes = {}
		encoder_x_y_units = zip(conf.encoder_x_y_units[:-1], conf.encoder_x_y_units[1:])
		for i, (n_in, n_out) in enumerate(encoder_x_y_units):
			encoder_x_y_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			encoder_x_y_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)
		encoder_x_y = SoftmaxEncoder(**encoder_x_y_attributes)
		encoder_x_y.n_layers = len(encoder_x_y_units)
		encoder_x_y.activation_function = conf.encoder_x_y_activation_function
		encoder_x_y.output_activation_function = conf.encoder_x_y_output_activation_function
		encoder_x_y.apply_dropout = conf.encoder_x_y_apply_dropout
		encoder_x_y.apply_batchnorm = conf.encoder_x_y_apply_batchnorm
		encoder_x_y.apply_batchnorm_to_input = conf.encoder_x_y_apply_batchnorm_to_input

		decoder_attributes = {}
		decoder_units = zip(conf.decoder_units[:-1], conf.decoder_units[1:])
		for i, (n_in, n_out) in enumerate(decoder_units):
			decoder_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			decoder_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)
		decoder = BernoulliDecoder(**decoder_attributes)
		decoder.n_layers = len(decoder_units)
		decoder.activation_function = conf.decoder_activation_function
		decoder.output_activation_function = conf.decoder_output_activation_function
		decoder.apply_dropout = conf.decoder_apply_dropout
		decoder.apply_batchnorm = conf.decoder_apply_batchnorm
		decoder.apply_batchnorm_to_input = conf.decoder_apply_batchnorm_to_input

		if conf.use_gpu:
			encoder_xy_z.to_gpu()
			encoder_x_y.to_gpu()
			decoder.to_gpu()
		return encoder_xy_z, encoder_x_y, decoder

	def encode_xy_z(self, x, y, test=False):
		input = concat_variables(x, y)
		return self.encoder_xy_z(input, test=test)

	def encode_x_y(self, x, test=False):
		return self.encoder_x_y(x, test=test)

	def decode_yz_x(self, z, y, test=False, output_pixel_value=False):
		input = concat_variables(z, y)
		return self.decoder(input, test=test, output_pixel_value=output_pixel_value)

	def loss_labeled(self, x, y, L=1, test=False):
		# Math:
		# -E_{q(z|x,y)}[logp(x|y,z) + logp(y)] + KL(q(z|x,y)||p(z))
		loss = 0
		z_mean, z_ln_var = self.encoder_xy_z(x, y, test=test)
		# E_{q(z|x,y)}[logp(x|y,z) + logp(y)]
		for l in xrange(L):
			# Sample z
			z = F.gaussian(z_mean, z_ln_var)
			# Decode
			x_expectation = self.decode_yz_x(z, y, test=test)
			# x is between -1 to 1 so we convert it to be between 0 to 1
			# logp(y) = log(1/ndim_y)
			reconstuction_loss = F.bernoulli_nll((x + 1.0) / 2.0, x_expectation) - math.log(1.0 / y.data.shape[1])
			loss += reconstuction_loss
		loss /= L * x.data.shape[0]
		# KL(q(z|x,y)||p(z))
		kld_regularization_loss = F.gaussian_kl_divergence(z_mean, z_ln_var)
		loss += kld_regularization_loss / x.data.shape[0]
		return loss

	def loss_unlabeled(self, x, L=1, test=False):
		pass

	def train(self, x, L=1, test=False):
		z_mean, z_ln_var = self.encoder(x, test=test, sample_output=False)
		loss = 0
		for l in xrange(L):
			# Sample z
			z = F.gaussian(z_mean, z_ln_var)
			# Decode
			x_expectation = self.decoder(z, test=test)
			# x is between -1 to 1 so we convert it to be between 0 to 1
			reconstuction_loss = F.bernoulli_nll((x + 1.0) / 2.0, x_expectation)
			loss += reconstuction_loss
		loss /= L * x.data.shape[0]
		# KL divergence
		kld_regularization_loss = F.gaussian_kl_divergence(z_mean, z_ln_var)
		loss += kld_regularization_loss / x.data.shape[0]

		self.zero_grads()
		loss.backward()
		self.update()

		if self.gpu:
			loss.to_cpu()
		return loss.data

class SoftmaxEncoder(chainer.Chain):
	def __init__(self, **layers):
		super(SoftmaxEncoder, self).__init__(**layers)
		self.activation_function = "tanh"
		self.output_activation_function = None
		self.apply_batchnorm_to_input = True
		self.apply_batchnorm = True
		self.apply_dropout = True

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def forward_one_step(self, x, test):
		f = activations[self.hidden_activation_function]
		chain = [x]

		for i in range(self.n_layers):
			u = chain[-1]
			if i == 0:
				if self.apply_batchnorm_to_input:
					u = getattr(self, "batchnorm_%d" % i)(u, test=test)
			else:
				if self.apply_batchnorm:
					u = getattr(self, "batchnorm_%d" % i)(u, test=test)
			u = getattr(self, "layer_%i" % i)(u)
			if i == self.n_layers - 1:
				if self.output_activation_function is None:
					output = u
				else:
					output = activations[self.output_activation_function](u)
			else:
				output = f(u)
				if self.apply_dropout:
					output = F.dropout(output, train=not test)
			chain.append(output)

		return chain[-1]

	def __call__(self, x, test=False, softmax=True):
		output = self.forward_one_step(x, test=test)
		if softmax:
			return F.softmax(output)
		return output

class GaussianEncoder(chainer.Chain):
	def __init__(self, **layers):
		super(GaussianEncoder, self).__init__(**layers)
		self.activation_function = "tanh"
		self.output_activation_function = None
		self.apply_batchnorm_to_input = True
		self.apply_batchnorm = True
		self.apply_dropout = True

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def forward_one_step(self, x, test=False, sample_output=True):
		f = activations[self.activation_function]

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
			u = getattr(self, "layer_mean_%i" % i)(u)
			if i == self.n_layers - 1:
				if self.output_activation_function is None:
					output = u
				else:
					output = activations[self.output_activation_function](u)
			else:
				output = f(u)
				if self.apply_dropout:
					output = F.dropout(output, train=not test)
			chain_mean.append(output)

			u = chain_variance[-1]
			if i == 0:
				if self.apply_batchnorm_to_input:
					u = getattr(self, "batchnorm_var_%i" % i)(u, test=test)
			else:
				if self.apply_batchnorm:
					u = getattr(self, "batchnorm_var_%i" % i)(u, test=test)
			u = getattr(self, "layer_var_%i" % i)(u)
			if i == self.n_layers - 1:
				if self.output_activation_function is None:
					output = u
				else:
					output = activations[self.output_activation_function](u)
			else:
				output = f(u)
				if self.apply_dropout:
					output = F.dropout(output, train=not test)
			chain_variance.append(output)

		mean = chain_mean[-1]
		# log(sigma^2)
		ln_var = chain_variance[-1]

		return mean, ln_var

	def __call__(self, x, test=False, sample_output=True):
		mean, ln_var = self.forward_one_step(x, test=test, sample_output=sample_output)
		if sample_output:
			return F.gaussian(mean, ln_var)
		return mean, ln_var

# Network structure is same as the Encoder
class GaussianDecoder(Encoder):

	def __call__(self, x, test=False, output_pixel_value=False):
		mean, ln_var = self.forward_one_step(x, test=test, sample_output=False)
		if output_pixel_value:
			return F.gaussian(mean, ln_var)
		return mean, ln_var

class BernoulliDecoder(SoftmaxEncoder):

	def __call__(self, x, test=False, output_pixel_value=False):
		output = self.forward_one_step(x, test=test)
		# Pixel value must be between -1 to 1
		if output_pixel_value:
			return (F.sigmoid(output) - 0.5) * 2.0
		return output

class Concat(function.Function):
	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(n_in == 2)
		a_type, b_type = in_types

		type_check.expect(
			a_type.dtype == np.float32,
			b_type.dtype == np.float32,
			a_type.ndim == 2,
			b_type.ndim == 2,
		)

	def forward(self, inputs):
		xp = cuda.get_array_module(inputs[0])
		v_a, v_b = inputs
		n_batch = v_a.shape[0]
		output = xp.empty((n_batch, v_a.shape[1] + v_b.shape[1]), dtype=xp.float32)
		output[:,:v_a.shape[1]] = v_a
		output[:,v_a.shape[1]:] = v_b
		return output,

	def backward(self, inputs, grad_outputs):
		v_a, v_b = inputs
		return grad_outputs[0][:,:v_a.shape[1]], grad_outputs[0][:,v_a.shape[1]:]

def concat_variables(v_a, v_b):
	return Concat()(v_a, v_b)