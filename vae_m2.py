# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six
from chainer import cuda, Variable, optimizers, serializers, function, optimizer
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
		self.ndim_z = 50

		# True : y = f(BN(Wx + b))
		# False: y = f(W*BN(x) + b)
		self.batchnorm_before_activation = True

		# gaussianmarg | gaussian
		self.type_pz = "gaussianmarg"
		self.type_qz = "gaussianmarg"

		# e.g.
		# {ndim_x + ndim_y} (input) -> 2000 -> 1000 -> 100 (output)
		# encoder_xy_z_hidden_units = [2000, 1000]
		self.encoder_xy_z_hidden_units = [500]
		self.encoder_xy_z_activation_function = "softplus"
		self.encoder_xy_z_apply_dropout = False
		self.encoder_xy_z_apply_batchnorm = True
		self.encoder_xy_z_apply_batchnorm_to_input = True

		self.encoder_x_y_hidden_units = [500]
		self.encoder_x_y_activation_function = "softplus"
		self.encoder_x_y_apply_dropout = False
		self.encoder_x_y_apply_batchnorm = True
		self.encoder_x_y_apply_batchnorm_to_input = True

		# e.g.
		# {ndim_z + ndim_y} (input) -> 2000 -> 1000 -> 100 (output)
		# decoder_hidden_units = [2000, 1000]
		self.decoder_hidden_units = [500]
		self.decoder_activation_function = "softplus"
		self.decoder_apply_dropout = False
		self.decoder_apply_batchnorm = True
		self.decoder_apply_batchnorm_to_input = True

		self.gpu_enabled = True
		self.learning_rate = 0.0003
		self.gradient_momentum = 0.9
		self.gradient_clipping = 5.0

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
			return
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
		# self.optimizer_encoder_xy_z.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_encoder_xy_z.add_hook(GradientClipping(conf.gradient_clipping))

		self.optimizer_encoder_x_y = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_encoder_x_y.setup(self.encoder_x_y)
		# self.optimizer_encoder_x_y.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_encoder_x_y.add_hook(GradientClipping(conf.gradient_clipping))

		self.optimizer_decoder = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_decoder.setup(self.decoder)
		# self.optimizer_decoder.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_decoder.add_hook(GradientClipping(conf.gradient_clipping))

		self.type_pz = conf.type_pz
		self.type_qz = conf.type_qz

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

	def update_classifier(self):
		self.optimizer_encoder_x_y.update()

	def encode_x_z(self, x, test=False):
		y = self.sample_x_y(x, argmax=False, test=test)
		z = self.encoder_xy_z(x, y, test=test)
		return z

	def encode_xy_z(self, x, y, test=False):
		z = self.encoder_xy_z(x, y, test=test)
		return z

	def decode_zy_x(self, z, y, test=False, apply_f=True):
		x = self.decoder(z, y, test=test, apply_f=apply_f)
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
		return F.sum(nll, axis=1)

	def gaussian_nll_keepbatch(self, x, mean, ln_var, clip=True):
		if clip:
			clip_min = math.log(0.001)
			clip_max = math.log(10)
			ln_var = F.clip(ln_var, clip_min, clip_max)
		x_prec = F.exp(-ln_var)
		x_diff = x - mean
		x_power = (x_diff * x_diff) * x_prec * 0.5
		return F.sum((math.log(2.0 * math.pi) + ln_var) * 0.5 + x_power, axis=1)

	def gaussian_kl_divergence_keepbatch(self, mean, ln_var):
		var = F.exp(ln_var)
		kld = F.sum(mean * mean + var - ln_var - 1, axis=1) * 0.5
		return kld

	def log_px_zy(self, x, z, y, test=False):
		if isinstance(self.decoder, BernoulliDecoder):
			# do not apply F.sigmoid to the output of the decoder
			raw_output = self.decoder(z, y, test=test, apply_f=False)
			negative_log_likelihood = self.bernoulli_nll_keepbatch(x, raw_output)
			log_px_zy = -negative_log_likelihood
		else:
			x_mean, x_ln_var = self.decoder(z, y, test=test, apply_f=False)
			negative_log_likelihood = self.gaussian_nll_keepbatch(x, x_mean, x_ln_var)
			log_px_zy = -negative_log_likelihood
		return log_px_zy

	def log_py(self, y, test=False):
		xp = self.xp
		num_types_of_label = y.data.shape[1]
		# prior p(y) expecting that all classes are evenly distributed
		constant = math.log(1.0 / num_types_of_label)
		log_py = xp.full((y.data.shape[0],), constant, xp.float32)
		return Variable(log_py)

	# this will not be used
	def log_pz(self, z, mean, ln_var, test=False):
		if self.type_pz == "gaussianmarg":
			# \int q(z)logp(z)dz = -(J/2)*log2pi - (1/2)*sum_{j=1}^{J} (mu^2 + var)
			# See Appendix B [Auto-Encoding Variational Bayes](http://arxiv.org/abs/1312.6114)
			log_pz = -0.5 * (math.log(2.0 * math.pi) + mean * mean + F.exp(ln_var))
		elif self.type_pz == "gaussian":
			log_pz = -0.5 * math.log(2.0 * math.pi) - 0.5 * z ** 2
		return F.sum(log_pz, axis=1)

	# this will not be used
	def log_qz_xy(self, z, mean, ln_var, test=False):
		if self.type_qz == "gaussianmarg":
			# \int q(z)logq(z)dz = -(J/2)*log2pi - (1/2)*sum_{j=1}^{J} (1 + logvar)
			# See Appendix B [Auto-Encoding Variational Bayes](http://arxiv.org/abs/1312.6114)
			log_qz_xy = -0.5 * F.sum((math.log(2.0 * math.pi) + 1 + ln_var), axis=1)
		elif self.type_qz == "gaussian":
			log_qz_xy = -self.gaussian_nll_keepbatch(z, mean, ln_var)
		return log_qz_xy

	def train(self, labeled_x, labeled_y, label_ids, unlabeled_x, test=False):
		loss, loss_labeled, loss_unlabeled = self.compute_lower_bound_loss(labeled_x, labeled_y, label_ids, unlabeled_x, test=test)
		self.zero_grads()
		loss.backward()
		self.update()

		if self.gpu:
			loss_labeled.to_cpu()
			if loss_unlabeled is not None:
				loss_unlabeled.to_cpu()

		if loss_unlabeled is None:
			return loss_labeled.data, 0

		return loss_labeled.data, loss_unlabeled.data

	# Extended objective eq.9
	def train_classification(self, labeled_x, label_ids, alpha=1.0, test=False):
		loss = alpha * self.compute_classification_loss(labeled_x, label_ids, test=test)
		self.zero_grads()
		loss.backward()
		self.update_classifier()
		if self.gpu:
			loss.to_cpu()
		return loss.data

	def train_jointly(self, labeled_x, labeled_y, label_ids, unlabeled_x, alpha=1.0, test=False):
		loss_lower_bound, loss_lb_labled, loss_lb_unlabled = self.compute_lower_bound_loss(labeled_x, labeled_y, label_ids, unlabeled_x, test=test)
		loss_classification = alpha * self.compute_classification_loss(labeled_x, label_ids, test=test)
		loss = loss_lower_bound + loss_classification
		self.zero_grads()
		loss.backward()
		self.update()
		if self.gpu:
			loss_lb_labled.to_cpu()
			if loss_lb_unlabled is not None:
				loss_lb_unlabled.to_cpu()
			loss_classification.to_cpu()

		if loss_lb_unlabled is None:
			return loss_lb_labled.data, 0, loss_classification.data

		return loss_lb_labled.data, loss_lb_unlabled.data, loss_classification.data

	def compute_lower_bound_loss(self, labeled_x, labeled_y, label_ids, unlabeled_x, test=False):

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
		z_mean_l, z_ln_var_l = self.encoder_xy_z(labeled_x, labeled_y, test=test, apply_f=False)
		z_l = self.encoder_xy_z(labeled_x, labeled_y, test=test)
		log_px_zy_l = self.log_px_zy(labeled_x, z_l, labeled_y, test=test)
		log_py_l = self.log_py(labeled_y, test=test)
		if False:
			log_pz_l = self.log_pz(z_l, z_mean_l, z_ln_var_l, test=test)
			log_qz_xy_l = self.log_qz_xy(z_l, z_mean_l, z_ln_var_l, test=test)
			lower_bound_l = lower_bound(log_px_zy_l, log_py_l, log_pz_l, log_qz_xy_l)
		else:
			lower_bound_l = log_px_zy_l + log_py_l - self.gaussian_kl_divergence_keepbatch(z_mean_l, z_ln_var_l)

		if batchsize_u > 0:
			### Lower bound for unlabeled data ###
			# To marginalize y, we repeat unlabeled x, and construct a target (batchsize_u * num_types_of_label) x num_types_of_label
			# Example of n-dimensional x and target matrix for a 3 class problem and batch_size=2.
			#         unlabeled_x_ext                 y_ext
			#  [[x0[0], x0[1], ..., x0[n]]         [[1, 0, 0]
			#   [x1[0], x1[1], ..., x1[n]]          [1, 0, 0]
			#   [x0[0], x0[1], ..., x0[n]]          [0, 1, 0]
			#   [x1[0], x1[1], ..., x1[n]]          [0, 1, 0]
			#   [x0[0], x0[1], ..., x0[n]]          [0, 0, 1]
			#   [x1[0], x1[1], ..., x1[n]]]         [0, 0, 1]]

			unlabeled_x_ext = xp.zeros((batchsize_u * num_types_of_label, unlabeled_x.data.shape[1]), dtype=xp.float32)
			y_ext = xp.zeros((batchsize_u * num_types_of_label, num_types_of_label), dtype=xp.float32)
			for n in xrange(num_types_of_label):
				y_ext[n * batchsize_u:(n + 1) * batchsize_u,n] = 1
				unlabeled_x_ext[n * batchsize_u:(n + 1) * batchsize_u] = unlabeled_x.data
			y_ext = Variable(y_ext)
			unlabeled_x_ext = Variable(unlabeled_x_ext)

			# Compute eq.6 -L(x,y) for unlabeled data
			z_mean_u_ext, z_mean_ln_var_u_ext = self.encoder_xy_z(unlabeled_x_ext, y_ext, test=test, apply_f=False)
			z_u_ext = F.gaussian(z_mean_u_ext, z_mean_ln_var_u_ext)
			log_px_zy_u = self.log_px_zy(unlabeled_x_ext, z_u_ext, y_ext, test=test)
			log_py_u = self.log_py(y_ext, test=test)
			if False:
				log_pz_u = self.log_pz(z_u_ext, z_mean_u_ext, z_mean_ln_var_u_ext, test=test)
				log_qz_xy_u = self.log_qz_xy(z_u_ext, z_mean_u_ext, z_mean_ln_var_u_ext, test=test)
				lower_bound_u = lower_bound(log_px_zy_u, log_py_u, log_pz_u, log_qz_xy_u)
			else:
				lower_bound_u = log_px_zy_u + log_py_u - self.gaussian_kl_divergence_keepbatch(z_mean_u_ext, z_mean_ln_var_u_ext)

			# Compute eq.7 sum_y{q(y|x){-L(x,y) + H(q(y|x))}}
			# Let LB(xn, y) be the lower bound for an input image xn and a label y (y = 0, 1, ..., 9).
			# Let bs be the batchsize.
			# 
			# lower_bound_u is a vector and it looks like...
			# [LB(x0,0), LB(x1,0), ..., LB(x_bs,0), LB(x0,1), LB(x1,1), ..., LB(x_bs,1), ..., LB(x0,9), LB(x1,9), ..., LB(x_bs,9)]
			# 
			# After reshaping. (axis 1 corresponds to label, axis 2 corresponds to batch)
			# [[LB(x0,0), LB(x1,0), ..., LB(x_bs,0)],
			#  [LB(x0,1), LB(x1,1), ..., LB(x_bs,1)],
			#                   .
			#                   .
			#                   .
			#  [LB(x0,9), LB(x1,9), ..., LB(x_bs,9)]]
			# 
			# After transposing. (axis 1 corresponds to batch)
			# [[LB(x0,0), LB(x0,1), ..., LB(x0,9)],
			#  [LB(x1,0), LB(x1,1), ..., LB(x1,9)],
			#                   .
			#                   .
			#                   .
			#  [LB(x_bs,0), LB(x_bs,1), ..., LB(x_bs,9)]]
			lower_bound_u = F.transpose(F.reshape(lower_bound_u, (num_types_of_label, batchsize_u)))
			
			y_distribution = self.encoder_x_y(unlabeled_x, test=test, softmax=True)
			lower_bound_u = y_distribution * (lower_bound_u - F.log(y_distribution + 1e-6))

			loss_labeled = -F.sum(lower_bound_l) / batchsize_l
			loss_unlabeled = -F.sum(lower_bound_u) / batchsize_u
			loss = loss_labeled + loss_unlabeled
		else:
			loss_unlabeled = None
			loss_labeled = -F.sum(lower_bound_l) / batchsize_l
			loss = loss_labeled

		return loss, loss_labeled, loss_unlabeled

	# Extended objective eq.9
	def compute_classification_loss(self, labeled_x, label_ids, test=False):
		y_distribution = self.encoder_x_y(labeled_x, softmax=False, test=test)
		batchsize = labeled_x.data.shape[0]
		num_types_of_label = y_distribution.data.shape[1]

		loss = F.softmax_cross_entropy(y_distribution, label_ids)
		return loss

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
		wscale = 0.1
		encoder_xy_z_attributes = {}
		encoder_xy_z_units = zip(conf.encoder_xy_z_hidden_units[:-1], conf.encoder_xy_z_hidden_units[1:])
		for i, (n_in, n_out) in enumerate(encoder_xy_z_units):
			encoder_xy_z_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			if conf.batchnorm_before_activation:
				encoder_xy_z_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				encoder_xy_z_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)
		encoder_xy_z_attributes["layer_merge_x"] = L.Linear(conf.ndim_x, conf.encoder_xy_z_hidden_units[0], wscale=wscale)
		encoder_xy_z_attributes["layer_merge_y"] = L.Linear(conf.ndim_y, conf.encoder_xy_z_hidden_units[0], wscale=wscale)
		encoder_xy_z_attributes["batchnorm_merge"] = L.BatchNormalization(conf.encoder_xy_z_hidden_units[0])
		encoder_xy_z_attributes["layer_output_mean"] = L.Linear(conf.encoder_xy_z_hidden_units[-1], conf.ndim_z, wscale=wscale)
		encoder_xy_z_attributes["layer_output_var"] = L.Linear(conf.encoder_xy_z_hidden_units[-1], conf.ndim_z, wscale=wscale)
		encoder_xy_z = GaussianEncoder(**encoder_xy_z_attributes)
		encoder_xy_z.n_layers = len(encoder_xy_z_units)
		encoder_xy_z.activation_function = conf.encoder_xy_z_activation_function
		encoder_xy_z.apply_dropout = conf.encoder_xy_z_apply_dropout
		encoder_xy_z.apply_batchnorm = conf.encoder_xy_z_apply_batchnorm
		encoder_xy_z.apply_batchnorm_to_input = conf.encoder_xy_z_apply_batchnorm_to_input
		encoder_xy_z.batchnorm_before_activation = conf.batchnorm_before_activation

		encoder_x_y_attributes = {}
		encoder_x_y_units = [(conf.ndim_x, conf.encoder_x_y_hidden_units[0])]
		encoder_x_y_units += zip(conf.encoder_x_y_hidden_units[:-1], conf.encoder_x_y_hidden_units[1:])
		encoder_x_y_units += [(conf.encoder_x_y_hidden_units[-1], conf.ndim_y)]
		for i, (n_in, n_out) in enumerate(encoder_x_y_units):
			encoder_x_y_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			if conf.batchnorm_before_activation:
				encoder_x_y_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				encoder_x_y_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)
		encoder_x_y = SoftmaxEncoder(**encoder_x_y_attributes)
		encoder_x_y.n_layers = len(encoder_x_y_units)
		encoder_x_y.activation_function = conf.encoder_x_y_activation_function
		encoder_x_y.apply_dropout = conf.encoder_x_y_apply_dropout
		encoder_x_y.apply_batchnorm = conf.encoder_x_y_apply_batchnorm
		encoder_x_y.apply_batchnorm_to_input = conf.encoder_x_y_apply_batchnorm_to_input
		encoder_x_y.batchnorm_before_activation = conf.batchnorm_before_activation

		decoder_attributes = {}
		decoder_units = zip(conf.decoder_hidden_units[:-1], conf.decoder_hidden_units[1:])
		for i, (n_in, n_out) in enumerate(decoder_units):
			decoder_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			if conf.batchnorm_before_activation:
				decoder_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				decoder_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)

		decoder_attributes["layer_merge_x"] = L.Linear(conf.ndim_z, conf.decoder_hidden_units[0], wscale=wscale)
		decoder_attributes["layer_merge_y"] = L.Linear(conf.ndim_y, conf.decoder_hidden_units[0], wscale=wscale)
		decoder_attributes["batchnorm_merge"] = L.BatchNormalization(conf.decoder_hidden_units[0])
		decoder_attributes["layer_output_mean"] = L.Linear(conf.decoder_hidden_units[-1], conf.ndim_x, wscale=wscale)
		decoder_attributes["layer_output_var"] = L.Linear(conf.decoder_hidden_units[-1], conf.ndim_x, wscale=wscale)
		decoder = GaussianDecoder(**decoder_attributes)
		decoder.n_layers = len(decoder_units)
		decoder.activation_function = conf.decoder_activation_function
		decoder.apply_dropout = conf.decoder_apply_dropout
		decoder.apply_batchnorm = conf.decoder_apply_batchnorm
		decoder.apply_batchnorm_to_input = conf.decoder_apply_batchnorm_to_input
		decoder.batchnorm_before_activation = conf.batchnorm_before_activation

		if conf.gpu_enabled:
			encoder_xy_z.to_gpu()
			encoder_x_y.to_gpu()
			decoder.to_gpu()
		return encoder_xy_z, encoder_x_y, decoder

class BernoulliM2VAE(VAE):

	def build(self, conf):
		wscale = 0.1
		encoder_xy_z_attributes = {}
		encoder_xy_z_units = zip(conf.encoder_xy_z_hidden_units[:-1], conf.encoder_xy_z_hidden_units[1:])
		for i, (n_in, n_out) in enumerate(encoder_xy_z_units):
			encoder_xy_z_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			if conf.batchnorm_before_activation:
				encoder_xy_z_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				encoder_xy_z_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)
		encoder_xy_z_attributes["layer_merge_x"] = L.Linear(conf.ndim_x, conf.encoder_xy_z_hidden_units[0], wscale=wscale)
		encoder_xy_z_attributes["layer_merge_y"] = L.Linear(conf.ndim_y, conf.encoder_xy_z_hidden_units[0], wscale=wscale)
		encoder_xy_z_attributes["batchnorm_merge"] = L.BatchNormalization(conf.encoder_xy_z_hidden_units[0])
		encoder_xy_z_attributes["layer_output_mean"] = L.Linear(conf.encoder_xy_z_hidden_units[-1], conf.ndim_z, wscale=wscale)
		encoder_xy_z_attributes["layer_output_var"] = L.Linear(conf.encoder_xy_z_hidden_units[-1], conf.ndim_z, wscale=wscale)
		encoder_xy_z = GaussianEncoder(**encoder_xy_z_attributes)
		encoder_xy_z.n_layers = len(encoder_xy_z_units)
		encoder_xy_z.activation_function = conf.encoder_xy_z_activation_function
		encoder_xy_z.apply_dropout = conf.encoder_xy_z_apply_dropout
		encoder_xy_z.apply_batchnorm = conf.encoder_xy_z_apply_batchnorm
		encoder_xy_z.apply_batchnorm_to_input = conf.encoder_xy_z_apply_batchnorm_to_input
		encoder_xy_z.batchnorm_before_activation = conf.batchnorm_before_activation

		encoder_x_y_attributes = {}
		encoder_x_y_units = [(conf.ndim_x, conf.encoder_x_y_hidden_units[0])]
		encoder_x_y_units += zip(conf.encoder_x_y_hidden_units[:-1], conf.encoder_x_y_hidden_units[1:])
		encoder_x_y_units += [(conf.encoder_x_y_hidden_units[-1], conf.ndim_y)]
		for i, (n_in, n_out) in enumerate(encoder_x_y_units):
			encoder_x_y_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			if conf.batchnorm_before_activation:
				encoder_x_y_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				encoder_x_y_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)
		encoder_x_y = SoftmaxEncoder(**encoder_x_y_attributes)
		encoder_x_y.n_layers = len(encoder_x_y_units)
		encoder_x_y.activation_function = conf.encoder_x_y_activation_function
		encoder_x_y.apply_dropout = conf.encoder_x_y_apply_dropout
		encoder_x_y.apply_batchnorm = conf.encoder_x_y_apply_batchnorm
		encoder_x_y.apply_batchnorm_to_input = conf.encoder_x_y_apply_batchnorm_to_input
		encoder_x_y.batchnorm_before_activation = conf.batchnorm_before_activation

		decoder_attributes = {}
		decoder_units = zip(conf.decoder_hidden_units[:-1], conf.decoder_hidden_units[1:])
		decoder_units += [(conf.decoder_hidden_units[-1], conf.ndim_x)]
		for i, (n_in, n_out) in enumerate(decoder_units):
			decoder_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			if conf.batchnorm_before_activation:
				decoder_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				decoder_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)
		decoder_attributes["layer_merge_z"] = L.Linear(conf.ndim_z, conf.decoder_hidden_units[0], wscale=wscale)
		decoder_attributes["layer_merge_y"] = L.Linear(conf.ndim_y, conf.decoder_hidden_units[0], wscale=wscale)
		decoder_attributes["batchnorm_merge"] = L.BatchNormalization(conf.decoder_hidden_units[0])
		decoder = BernoulliDecoder(**decoder_attributes)
		decoder.n_layers = len(decoder_units)
		decoder.activation_function = conf.decoder_activation_function
		decoder.apply_dropout = conf.decoder_apply_dropout
		decoder.apply_batchnorm = conf.decoder_apply_batchnorm
		decoder.apply_batchnorm_to_input = conf.decoder_apply_batchnorm_to_input
		decoder.batchnorm_before_activation = conf.batchnorm_before_activation

		if conf.gpu_enabled:
			encoder_xy_z.to_gpu()
			encoder_x_y.to_gpu()
			decoder.to_gpu()
		return encoder_xy_z, encoder_x_y, decoder

class SoftmaxEncoder(chainer.Chain):
	def __init__(self, **layers):
		super(SoftmaxEncoder, self).__init__(**layers)
		self.activation_function = "softplus"
		self.apply_batchnorm_to_input = True
		self.apply_batchnorm = True
		self.apply_dropout = False
		self.batchnorm_before_activation = True

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def forward_one_step(self, x, test):
		f = activations[self.activation_function]
		chain = [x]

		for i in range(self.n_layers):
			u = chain[-1]
			if self.batchnorm_before_activation:
				u = getattr(self, "layer_%i" % i)(u)
			if i == 0:
				if self.apply_batchnorm_to_input:
					u = getattr(self, "batchnorm_%d" % i)(u, test=test)
			elif i == self.n_layers - 1:
				if self.apply_batchnorm and self.batchnorm_before_activation == False:
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

	def __call__(self, x, test=False, softmax=True):
		output = self.forward_one_step(x, test=test)
		if softmax:
			return F.softmax(output)
		return output

class GaussianEncoder(chainer.Chain):
	def __init__(self, **layers):
		super(GaussianEncoder, self).__init__(**layers)
		self.activation_function = "softplus"
		self.apply_batchnorm_to_input = True
		self.apply_batchnorm = True
		self.apply_dropout = False
		self.batchnorm_before_activation = True

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def forward_one_step(self, x, y, test=False, apply_f=True):
		f = activations[self.activation_function]

		if self.apply_batchnorm_to_input:
			if self.batchnorm_before_activation:
				merged_input = f(self.batchnorm_merge(self.layer_merge_x(x) + self.layer_merge_y(y), test=test))
			else:
				merged_input = f(self.layer_merge_x(self.batchnorm_merge(x, test=test)) + self.layer_merge_y(y))
		else:
			merged_input = f(self.layer_merge_x(x) + self.layer_merge_y(y))

		chain = [merged_input]

		# Hidden
		for i in range(self.n_layers):
			u = chain[-1]
			if batchnorm_before_activation:
				u = getattr(self, "layer_%i" % i)(u)
			if self.apply_batchnorm:
				u = getattr(self, "batchnorm_%d" % i)(u, test=test)
			if batchnorm_before_activation == False:
				u = getattr(self, "layer_%i" % i)(u)
			output = f(u)
			if self.apply_dropout:
				output = F.dropout(output, train=not test)
			chain.append(output)

		u = chain[-1]
		mean = self.layer_output_mean(u)

		# log(sd^2)
		u = chain[-1]
		ln_var = self.layer_output_var(u)

		return mean, ln_var

	def __call__(self, x, y, test=False, apply_f=True):
		mean, ln_var = self.forward_one_step(x, y, test=test, apply_f=apply_f)
		if apply_f:
			return F.gaussian(mean, ln_var)
		return mean, ln_var

# Network structure is same as the GaussianEncoder
class GaussianDecoder(GaussianEncoder):

	def __call__(self, z, y, test=False, apply_f=False):
		mean, ln_var = self.forward_one_step(z, y, test=test, apply_f=False)
		if apply_f:
			return F.gaussian(mean, ln_var)
		return mean, ln_var

class BernoulliDecoder(SoftmaxEncoder):

	def forward_one_step(self, z, y, test):
		f = activations[self.activation_function]

		if self.apply_batchnorm_to_input:
			if self.batchnorm_before_activation:
				merged_input = f(self.batchnorm_merge(self.layer_merge_z(z) + self.layer_merge_y(y), test=test))
			else:
				merged_input = f(self.layer_merge_z(self.batchnorm_merge(z, test=test)) + self.layer_merge_y(y))
		else:
			merged_input = f(self.layer_merge_z(z) + self.layer_merge_y(y))

		chain = [merged_input]

		# Hidden
		for i in range(self.n_layers):
			u = chain[-1]
			if self.batchnorm_before_activation:
				u = getattr(self, "layer_%i" % i)(u)
			if i == self.n_layers - 1:
				if self.apply_batchnorm and self.batchnorm_before_activation == False:
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

	def __call__(self, z, y, test=False, apply_f=False):
		output = self.forward_one_step(z, y, test=test)
		if apply_f:
			return F.sigmoid(output)
		return output