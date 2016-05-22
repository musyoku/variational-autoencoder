# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six
from chainer import cuda, Variable, optimizers, serializers, function
from chainer.utils import type_check
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

		# e.g.
		# ndim_x + ndim_y(input) -> 2000 -> 1000 -> 100 (output)
		# encoder_xy_z_hidden_units = [2000, 1000]
		self.encoder_xy_z_hidden_units = [600, 600]
		self.encoder_xy_z_activation_function = "softplus"
		self.encoder_xy_z_apply_dropout = False
		self.encoder_xy_z_apply_batchnorm = False
		self.encoder_xy_z_apply_batchnorm_to_input = False

		self.encoder_x_y_hidden_units = [600, 600]
		self.encoder_x_y_activation_function = "softplus"
		self.encoder_x_y_apply_dropout = False
		self.encoder_x_y_apply_batchnorm = False
		self.encoder_x_y_apply_batchnorm_to_input = False

		# e.g.
		# ndim_z + ndim_y(input) -> 2000 -> 1000 -> 100 (output)
		# decoder_hidden_units = [2000, 1000]
		self.decoder_hidden_units = [600, 600]
		self.decoder_activation_function = "softplus"
		self.decoder_apply_dropout = False
		self.decoder_apply_batchnorm = False
		self.decoder_apply_batchnorm_to_input = False

		self.gpu_enabled = True
		self.learning_rate = 0.0003
		self.gradient_momentum = 0.9

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
		# self.optimizer_encoder_xy_z.add_hook(GradientClipping(1.0))

		self.optimizer_encoder_x_y = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_encoder_x_y.setup(self.encoder_x_y)
		# self.optimizer_encoder_x_y.add_hook(GradientClipping(1.0))

		self.optimizer_decoder = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_decoder.setup(self.decoder)
		# self.optimizer_decoder.add_hook(GradientClipping(1.0))

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

	def encode_x_z(self, x, test=False):
		y = self.sample_x_y(x, argmax=False, test=test)
		z = self.encoder_xy_z(x, y, test=test)
		return z

	def encode_xy_z(self, x, test=False):
		z = self.encoder_xy_z(x, y, test=test)
		return z

	def decode_zy_x(self, z, y, test=False, output_pixel_value=True):
		x = self.decoder(z, y, test=False, output_pixel_value=output_pixel_value)
		return x

	def sample_x_y(self, x, argmax=False, test=False):
		batchsize = x.data.shape[0]
		y_distribution = self.encoder_x_y(x, test=test, softmax=True).data
		n_labels = y_distribution.shape[1]
		if self.gpu:
			y_distribution = cuda.to_cpu(y_distribution)
		sampled_y = np.zeros((batchsize, n_labels), dtype=np.float32)
		if argmax:
			args = np.argmax(y_distribution, axis=1)
			for b in xrange(batchsize):
				sampled_y[b, args[b]] = 1
		else:
			for b in xrange(batchsize):
				label_id = np.random.choice(np.arange(n_labels), p=y_distribution[b])
				sampled_y[b, label_id] = 1
		sampled_y = Variable(sampled_y)
		if self.gpu:
			sampled_y.to_gpu()
		return sampled_y

	def sample_x_label(self, x, argmax=True, test=False):
		batchsize = x.data.shape[0]
		y_distribution = self.encoder_x_y(x, test=test, softmax=True).data
		n_labels = y_distribution.shape[1]
		if self.gpu:
			y_distribution = cuda.to_cpu(y_distribution)
		if argmax:
			sampled_label = np.argmax(y_distribution, axis=1)
		else:
			sampled_label = np.zeros((batchsize,), dtype=np.int32)
			labels = np.arange(n_labels)
			for b in xrange(batchsize):
				label_id = np.random.choice(labels, p=y_distribution[b])
				sampled_label[b] = 1
		return sampled_label

	def bernoulli_nll_keepbatch(self, x, y):
		nll = F.softplus(y) - x * y
		return F.sum(nll, axis=1) / x.data.shape[1]

	def gaussian_nll_keepbatch(self, x, mean, ln_var):
		x_prec = F.exp(-ln_var)
		x_diff = x - mean
		x_power = (x_diff * x_diff) * x_prec * 0.5
		return F.sum((math.log(2.0 * math.pi) + ln_var) * 0.5 + x_power, axis=1) / x.data.shape[1]

	def gaussian_kl_divergence_keepbatch(self, mean, ln_var):
		var = F.exp(ln_var)
		kld = F.sum(mean * mean + var - ln_var - 1, axis=1) * 0.5 / mean.data.shape[1]
		return kld

	def log_px_zy(self, x, z, y, test=False):
		if isinstance(self.decoder, BernoulliDecoder):
			# do not apply F.sigmoid to the output of the decoder
			x_expectation = self.decoder(z, y, test=test, output_pixel_value=False)
			negative_log_likelihood = self.bernoulli_nll_keepbatch((x + 1.0) / 2.0, x_expectation)
			log_px_zy = -negative_log_likelihood
		else:
			x_mean, x_ln_var = self.decoder(z, y, test=test, output_pixel_value=False)
			negative_log_likelihood = self.gaussian_nll_keepbatch(x, x_mean, x_ln_var)
			log_px_zy = -negative_log_likelihood
		return log_px_zy

	def log_py(self, y, test=False):
		xp = self.xp
		# prior p(y) expecting that all classes are evenly distributed
		constant = math.log(1.0 / y.data.shape[1])
		log_py = xp.full((y.data.shape[0],), constant, xp.float32)
		return Variable(log_py)

	def log_pz(self, z, test=False):
		constant = -0.5 * math.log(2.0 * math.pi)
		log_pz = constant - 0.5 * z ** 2
		return F.sum(log_pz, axis=1)

	def log_qz_xy(self, x, y, z, test=False):
		z_mean, z_ln_var = self.encoder_xy_z(x, y, test=test, sample_output=False)
		negative_log_likelihood = self.gaussian_nll_keepbatch(z, z_mean, z_ln_var)
		log_qz_xy = -negative_log_likelihood
		return log_qz_xy

	def log_qy_x(self, x, y, test=False):
		y_expectation = self.encoder_x_y(x, test=False, softmax=True)
		log_qy_x = y * F.log(y_expectation + 1e-6)
		return log_qy_x

	def train(self, labeled_x, labeled_y, label_ids, unlabeled_x, test=False):

		def lower_bound(log_px_zy, log_py, log_pz, log_qz_xy):
			lb = log_px_zy + log_py + log_pz - log_qz_xy
			return lb

		# _l: labeled
		# _u: unlabeled
		batchsize_l = labeled_x.data.shape[0]
		batchsize_u = unlabeled_x.data.shape[0]
		num_types_of_label = labeled_y.data.shape[1]
		xp = self.xp

		### Lower bound for labeled data ###
		# Compute eq.6 -L(x,y)
		z_l = self.encoder_xy_z(labeled_x, labeled_y, test=test)
		log_px_zy_l = self.log_px_zy(labeled_x, z_l, labeled_y, test=test)
		log_py_l = self.log_py(labeled_y, test=test)
		log_pz_l = self.log_pz(z_l, test=test)
		log_qz_xy_l = self.log_qz_xy(labeled_x, labeled_y, z_l, test=test)
		lower_bound_l = lower_bound(log_px_zy_l, log_py_l, log_pz_l, log_qz_xy_l)

		### Lower bound for unlabeled data ###
		# To marginalize y, we repeat unlabeled x, and construct a target (batchsize_u * num_types_of_label) x num_types_of_label
		# Example of input and target matrix for a 3 class problem and batch_size=2.
		#            unlabeled_x_ext                  y_ext
		#  [[x[0,0], x[0,1], ..., x[0,n_x]]         [[1, 0, 0]
		#   [x[1,0], x[1,1], ..., x[1,n_x]]          [1, 0, 0]
		#   [x[0,0], x[0,1], ..., x[0,n_x]]          [0, 1, 0]
		#   [x[1,0], x[1,1], ..., x[1,n_x]]          [0, 1, 0]
		#   [x[0,0], x[0,1], ..., x[0,n_x]]          [0, 0, 1]
		#   [x[1,0], x[1,1], ..., x[1,n_x]]]         [0, 0, 1]]
		# We thunk Lars Maaloe for this idea.
		# See https://github.com/larsmaaloee/auxiliary-deep-generative-models

		unlabeled_x_ext = xp.zeros((batchsize_u * num_types_of_label, unlabeled_x.data.shape[1]), dtype=xp.float32)
		y_ext = xp.zeros((batchsize_u * num_types_of_label, num_types_of_label), dtype=xp.float32)
		for n in xrange(num_types_of_label):
			y_ext[n * batchsize_u:(n + 1) * batchsize_u,n] = 1
			unlabeled_x_ext[n * batchsize_u:(n + 1) * batchsize_u] = unlabeled_x.data
		y_ext = Variable(y_ext)
		unlabeled_x_ext = Variable(unlabeled_x_ext)

		# Compute eq.6 -L(x,y) for unlabeled data
		z_u = self.encoder_xy_z(unlabeled_x_ext, y_ext, test=test)
		log_px_zy_u = self.log_px_zy(unlabeled_x_ext, z_u, y_ext, test=test)
		log_py_u = self.log_py(y_ext, test=test)
		log_pz_u = self.log_pz(z_u, test=test)
		log_qz_xy_u = self.log_qz_xy(unlabeled_x_ext, y_ext, z_u, test=test)
		lower_bound_u = lower_bound(log_px_zy_u, log_py_u, log_pz_u, log_qz_xy_u)

		# Compute eq.7 sum_y{q(y|x){-L(x,y) + H(q(y|x))}}
		# LB(x, y) represents lower bound for an input image x and a label y (y = 0, 1, ..., 9)
		# 
		# lower_bound_u is a vector contains...
		# [LB(x0,0), LB(x1,0), ..., LB(x_n,0), LB(x0,1), LB(x1,1), ..., LB(x_n,1), ..., LB(x0,9), LB(x1,9), ..., LB(x_n,9)]
		# 
		# After reshaping. (axis 1 corresponds to label, axis 2 corresponds to batch)
		# [[LB(x0,0), LB(x1,0), ..., LB(x_n,0)],
		#  [LB(x0,1), LB(x1,1), ..., LB(x_n,1)],
		#                           .
		#                           .
		#                           .
		#  [LB(x0,9), LB(x1,9), ..., LB(x_n,9)]]
		# 
		# After transposing. (axis 1 corresponds to batch)
		# [[LB(x0,0), LB(x0,1), ..., LB(x0,9)],
		#  [LB(x1,0), LB(x1,1), ..., LB(x1,9)],
		#                           .
		#                           .
		#                           .
		#  [LB(x_n,0), LB(x_n,1), ..., LB(x_n,9)]]
		y_distribution = self.encoder_x_y(unlabeled_x, test=test, softmax=True)
		lower_bound_u = F.transpose(F.reshape(lower_bound_u, (num_types_of_label, batchsize_u)))
		lower_bound_u = y_distribution * (lower_bound_u - F.log(y_distribution + 1e-6))

		loss_labeled = -F.sum(lower_bound_l) / batchsize_l
		loss_unlabeled = -F.sum(lower_bound_u) / batchsize_u
		loss = loss_labeled + loss_unlabeled

		val = cuda.to_cpu(loss.data)
		if val != val:
			raise Exception("You have encountered NaN!")

		self.zero_grads()
		loss.backward()
		self.update()

		if self.gpu:
			loss_labeled.to_cpu()
			loss_unlabeled.to_cpu()
		return loss_labeled.data, loss_unlabeled.data

	# Extended objective eq.9
	def train_classification(self, labeled_x, label_ids, alpha=1.0, test=False):
		y_distribution = self.encoder_x_y(labeled_x, softmax=False, test=test)
		batchsize = labeled_x.data.shape[0]
		num_types_of_label = y_distribution.data.shape[1]

		loss_classifier = alpha * F.softmax_cross_entropy(y_distribution, label_ids)
		self.zero_grads()
		loss_classifier.backward()
		self.update()
		if self.gpu:
			loss_classifier.to_cpu()
		return loss_classifier.data

	def load(self, dir=None):
		if dir is None:
			raise Exception()
		for attr in vars(self):
			prop = getattr(self, attr)
			if isinstance(prop, chainer.Chain) or isinstance(prop, chainer.optimizer.GradientMethod):
				filename = dir + "/%s_%s.hdf5" % (self.name, attr)
				if os.path.isfile(filename):
					print "loading",  filename
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
		wscale = 1.0
		encoder_xy_z_attributes = {}
		encoder_xy_z_units = zip(conf.encoder_xy_z_hidden_units[:-1], conf.encoder_xy_z_hidden_units[1:])
		encoder_xy_z_units += [(conf.encoder_x_y_hidden_units[-1], conf.ndim_z)]
		for i, (n_in, n_out) in enumerate(encoder_xy_z_units):
			encoder_xy_z_attributes["layer_mean_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			encoder_xy_z_attributes["batchnorm_mean_%i" % i] = L.BatchNormalization(n_out)
			encoder_xy_z_attributes["layer_var_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			encoder_xy_z_attributes["batchnorm_var_%i" % i] = L.BatchNormalization(n_out)
		encoder_xy_z_attributes["layer_mean_merge_x"] = L.Linear(conf.ndim_x, conf.encoder_xy_z_hidden_units[0])
		encoder_xy_z_attributes["layer_mean_merge_y"] = L.Linear(conf.ndim_y, conf.encoder_xy_z_hidden_units[0])
		encoder_xy_z_attributes["batchnorm_mean_merge"] = L.BatchNormalization(conf.encoder_xy_z_hidden_units[0])
		encoder_xy_z_attributes["layer_var_merge_x"] = L.Linear(conf.ndim_x, conf.encoder_xy_z_hidden_units[0])
		encoder_xy_z_attributes["layer_var_merge_y"] = L.Linear(conf.ndim_y, conf.encoder_xy_z_hidden_units[0])
		encoder_xy_z_attributes["batchnorm_var_merge"] = L.BatchNormalization(conf.encoder_xy_z_hidden_units[0])
		encoder_xy_z = GaussianEncoder(**encoder_xy_z_attributes)
		encoder_xy_z.n_layers = len(encoder_xy_z_units)
		encoder_xy_z.activation_function = conf.encoder_xy_z_activation_function
		encoder_xy_z.apply_dropout = conf.encoder_xy_z_apply_dropout
		encoder_xy_z.apply_batchnorm = conf.encoder_xy_z_apply_batchnorm
		encoder_xy_z.apply_batchnorm_to_input = conf.encoder_xy_z_apply_batchnorm_to_input

		encoder_x_y_attributes = {}
		encoder_x_y_units = [(conf.ndim_x, conf.encoder_x_y_hidden_units[0])]
		encoder_x_y_units += zip(conf.encoder_x_y_hidden_units[:-1], conf.encoder_x_y_hidden_units[1:])
		encoder_x_y_units += [(conf.encoder_x_y_hidden_units[-1], conf.ndim_y)]
		for i, (n_in, n_out) in enumerate(encoder_x_y_units):
			encoder_x_y_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			encoder_x_y_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
		encoder_x_y = SoftmaxEncoder(**encoder_x_y_attributes)
		encoder_x_y.n_layers = len(encoder_x_y_units)
		encoder_x_y.activation_function = conf.encoder_x_y_activation_function
		encoder_x_y.apply_dropout = conf.encoder_x_y_apply_dropout
		encoder_x_y.apply_batchnorm = conf.encoder_x_y_apply_batchnorm
		encoder_x_y.apply_batchnorm_to_input = conf.encoder_x_y_apply_batchnorm_to_input

		decoder_attributes = {}
		decoder_units = zip(conf.decoder_hidden_units[:-1], conf.decoder_hidden_units[1:])
		decoder_units += [(conf.decoder_hidden_units[-1], conf.ndim_x)]
		for i, (n_in, n_out) in enumerate(decoder_units):
			decoder_attributes["layer_mean_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			decoder_attributes["batchnorm_mean_%i" % i] = L.BatchNormalization(n_out)
			decoder_attributes["layer_var_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			decoder_attributes["batchnorm_var_%i" % i] = L.BatchNormalization(n_out)

		# Note: GaussianDecoder is the same as GaussianEncoder (it takes x and y)
		decoder_attributes["layer_mean_merge_x"] = L.Linear(conf.ndim_z, conf.decoder_hidden_units[0], wscale=wscale)
		decoder_attributes["layer_mean_merge_y"] = L.Linear(conf.ndim_y, conf.decoder_hidden_units[0], wscale=wscale)
		decoder_attributes["batchnorm_mean_merge"] = L.BatchNormalization(conf.decoder_hidden_units[0])
		decoder_attributes["layer_var_merge_x"] = L.Linear(conf.ndim_z, conf.decoder_hidden_units[0], wscale=wscale)
		decoder_attributes["layer_var_merge_y"] = L.Linear(conf.ndim_y, conf.decoder_hidden_units[0], wscale=wscale)
		decoder_attributes["batchnorm_var_merge"] = L.BatchNormalization(conf.decoder_hidden_units[0])
		decoder = GaussianDecoder(**decoder_attributes)
		decoder.n_layers = len(decoder_units)
		decoder.activation_function = conf.decoder_activation_function
		decoder.apply_dropout = conf.decoder_apply_dropout
		decoder.apply_batchnorm = conf.decoder_apply_batchnorm
		decoder.apply_batchnorm_to_input = conf.decoder_apply_batchnorm_to_input

		if conf.gpu_enabled:
			encoder_xy_z.to_gpu()
			encoder_x_y.to_gpu()
			decoder.to_gpu()
		return encoder_xy_z, encoder_x_y, decoder

class BernoulliM2VAE(VAE):

	def build(self, conf):
		wscale = 1.0
		encoder_xy_z_attributes = {}
		encoder_xy_z_units = zip(conf.encoder_xy_z_hidden_units[:-1], conf.encoder_xy_z_hidden_units[1:])
		encoder_xy_z_units += [(conf.encoder_xy_z_hidden_units[-1], conf.ndim_z)]
		for i, (n_in, n_out) in enumerate(encoder_xy_z_units):
			encoder_xy_z_attributes["layer_mean_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			encoder_xy_z_attributes["batchnorm_mean_%i" % i] = L.BatchNormalization(n_out)
			encoder_xy_z_attributes["layer_var_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			encoder_xy_z_attributes["batchnorm_var_%i" % i] = L.BatchNormalization(n_out)
		encoder_xy_z_attributes["layer_mean_merge_x"] = L.Linear(conf.ndim_x, conf.encoder_xy_z_hidden_units[0], wscale=wscale)
		encoder_xy_z_attributes["batchnorm_mean_merge"] = L.BatchNormalization(conf.encoder_xy_z_hidden_units[0])
		encoder_xy_z_attributes["layer_var_merge_x"] = L.Linear(conf.ndim_x, conf.encoder_xy_z_hidden_units[0], wscale=wscale)
		encoder_xy_z_attributes["batchnorm_var_merge"] = L.BatchNormalization(conf.encoder_xy_z_hidden_units[0])
		encoder_xy_z_attributes["layer_mean_merge_y"] = L.Linear(conf.ndim_y, conf.encoder_xy_z_hidden_units[0], wscale=wscale)
		encoder_xy_z_attributes["layer_var_merge_y"] = L.Linear(conf.ndim_y, conf.encoder_xy_z_hidden_units[0], wscale=wscale)
		encoder_xy_z = GaussianEncoder(**encoder_xy_z_attributes)
		encoder_xy_z.n_layers = len(encoder_xy_z_units)
		encoder_xy_z.activation_function = conf.encoder_xy_z_activation_function
		encoder_xy_z.apply_dropout = conf.encoder_xy_z_apply_dropout
		encoder_xy_z.apply_batchnorm = conf.encoder_xy_z_apply_batchnorm
		encoder_xy_z.apply_batchnorm_to_input = conf.encoder_xy_z_apply_batchnorm_to_input

		encoder_x_y_attributes = {}
		encoder_x_y_units = [(conf.ndim_x, conf.encoder_x_y_hidden_units[0])]
		encoder_x_y_units += zip(conf.encoder_x_y_hidden_units[:-1], conf.encoder_x_y_hidden_units[1:])
		encoder_x_y_units += [(conf.encoder_x_y_hidden_units[-1], conf.ndim_y)]
		for i, (n_in, n_out) in enumerate(encoder_x_y_units):
			encoder_x_y_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			encoder_x_y_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
		encoder_x_y = SoftmaxEncoder(**encoder_x_y_attributes)
		encoder_x_y.n_layers = len(encoder_x_y_units)
		encoder_x_y.activation_function = conf.encoder_x_y_activation_function
		encoder_x_y.apply_dropout = conf.encoder_x_y_apply_dropout
		encoder_x_y.apply_batchnorm = conf.encoder_x_y_apply_batchnorm
		encoder_x_y.apply_batchnorm_to_input = conf.encoder_x_y_apply_batchnorm_to_input

		decoder_attributes = {}
		decoder_units = zip(conf.decoder_hidden_units[:-1], conf.decoder_hidden_units[1:])
		decoder_units += [(conf.decoder_hidden_units[-1], conf.ndim_x)]
		for i, (n_in, n_out) in enumerate(decoder_units):
			decoder_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			decoder_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
		decoder_attributes["layer_merge_z"] = L.Linear(conf.ndim_z, conf.decoder_hidden_units[0], wscale=wscale)
		decoder_attributes["batchnorm_merge"] = L.BatchNormalization(conf.decoder_hidden_units[0])
		decoder_attributes["layer_merge_y"] = L.Linear(conf.ndim_y, conf.decoder_hidden_units[0], wscale=wscale)
		decoder = BernoulliDecoder(**decoder_attributes)
		decoder.n_layers = len(decoder_units)
		decoder.activation_function = conf.decoder_activation_function
		decoder.apply_dropout = conf.decoder_apply_dropout
		decoder.apply_batchnorm = conf.decoder_apply_batchnorm
		decoder.apply_batchnorm_to_input = conf.decoder_apply_batchnorm_to_input

		if conf.gpu_enabled:
			encoder_xy_z.to_gpu()
			encoder_x_y.to_gpu()
			decoder.to_gpu()
		return encoder_xy_z, encoder_x_y, decoder

class SoftmaxEncoder(chainer.Chain):
	def __init__(self, **layers):
		super(SoftmaxEncoder, self).__init__(**layers)
		self.activation_function = "softplus"
		self.apply_batchnorm_to_input = False
		self.apply_batchnorm = False
		self.apply_dropout = False

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def forward_one_step(self, x, test):
		f = activations[self.activation_function]
		chain = [x]

		for i in range(self.n_layers):
			u = chain[-1]
			u = getattr(self, "layer_%i" % i)(u)
			if i == 0:
				if self.apply_batchnorm_to_input:
					u = getattr(self, "batchnorm_%d" % i)(u, test=test)
			else:
				if self.apply_batchnorm:
					u = getattr(self, "batchnorm_%d" % i)(u, test=test)
			if i == self.n_layers - 1:
				output = u
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
		self.activation_function = "softplus"
		self.apply_batchnorm_to_input = False
		self.apply_batchnorm = False
		self.apply_dropout = False

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def forward_one_step(self, x, y, test=False, sample_output=True):
		f = activations[self.activation_function]

		if self.apply_batchnorm_to_input:
			merged_input_mean = f(self.batchnorm_mean_merge(self.layer_mean_merge_x(x) + self.layer_mean_merge_y(y), test=test))
			merged_input_var = f(self.batchnorm_var_merge(self.layer_var_merge_x(x) + self.layer_var_merge_y(y), test=test))
		else:
			merged_input_mean = f(self.layer_mean_merge_x(x) + self.layer_mean_merge_y(y))
			merged_input_var = f(self.layer_var_merge_x(x) + self.layer_var_merge_y(y))

		chain_mean = [merged_input_mean]
		chain_variance = [merged_input_var]

		# Hidden
		for i in range(self.n_layers):
			# mean
			u = chain_mean[-1]
			u = getattr(self, "layer_mean_%i" % i)(u)
			if self.apply_batchnorm:
				u = getattr(self, "batchnorm_mean_%d" % i)(u, test=test)
			if i == self.n_layers - 1:
				output = u
			else:
				output = f(u)
				if self.apply_dropout:
					output = F.dropout(output, train=not test)
			chain_mean.append(output)

			# variance
			u = chain_variance[-1]
			u = getattr(self, "layer_var_%i" % i)(u)
			if self.apply_batchnorm:
				u = getattr(self, "batchnorm_var_%i" % i)(u, test=test)
			if i == self.n_layers - 1:
				output = u
			else:
				output = f(u)
				if self.apply_dropout:
					output = F.dropout(output, train=not test)
			chain_variance.append(output)

		mean = chain_mean[-1]
		# log(variance)
		ln_var = chain_variance[-1]

		return mean, ln_var

	def __call__(self, x, y, test=False, sample_output=True):
		mean, ln_var = self.forward_one_step(x, y, test=test, sample_output=sample_output)
		if sample_output:
			return F.gaussian(mean, ln_var)
		return mean, ln_var

# Network structure is same as the GaussianEncoder
class GaussianDecoder(GaussianEncoder):

	def __call__(self, z, y, test=False, output_pixel_value=False):
		mean, ln_var = self.forward_one_step(z, y, test=test, sample_output=False)
		if output_pixel_value:
			return F.gaussian(mean, ln_var)
		return mean, ln_var

class BernoulliDecoder(SoftmaxEncoder):

	def forward_one_step(self, z, y, test):
		f = activations[self.activation_function]

		if self.apply_batchnorm_to_input:
			merged_input = f(self.batchnorm_merge(self.layer_merge_z(z) + self.layer_merge_y(y), test=test))
		else:
			merged_input = f(self.layer_merge_z(z) + self.layer_merge_y(y))

		chain = [merged_input]

		for i in range(self.n_layers):
			u = chain[-1]
			u = getattr(self, "layer_%i" % i)(u)
			if self.apply_batchnorm:
				u = getattr(self, "batchnorm_%d" % i)(u, test=test)
			if i == self.n_layers - 1:
				output = u
			else:
				output = f(u)
				if self.apply_dropout:
					output = F.dropout(output, train=not test)
			chain.append(output)

		return chain[-1]

	def __call__(self, z, y, test=False, output_pixel_value=False):
		output = self.forward_one_step(z, y, test=test)
		if output_pixel_value:
			return (F.sigmoid(output) - 0.5) * 2.0
		return output