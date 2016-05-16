import unittest
import numpy
import chainer
from chainer import cuda
from chainer import functions
from chainer.functions.connection import linear
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
import os, sys, time
sys.path.append(os.path.split(os.getcwd())[0])
import vae_m2

class TestVectorize(unittest.TestCase):

	def setUp(self):
		self.vector = numpy.random.uniform(1, 1, (3, 4)).astype(numpy.float32)
		self.scalar = numpy.random.uniform(1, 1, (3, 1)).astype(numpy.float32)
		self.gy = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
		self.y = self.vector * self.scalar

	def check_forward(self, vector, scalar, y_expect):
		vector = chainer.Variable(vector)
		scalar = chainer.Variable(scalar)
		y = vae_m2.multiply(vector, scalar)
		gradient_check.assert_allclose(y_expect, y.data)

	@condition.retry(3)
	def test_forward_cpu(self):
		self.check_forward(self.vector, self.scalar, self.y)

	@attr.gpu
	@condition.retry(3)
	def test_forward_gpu(self):
		self.check_forward(cuda.to_gpu(self.vector), cuda.to_gpu(self.scalar), cuda.to_gpu(self.vector * self.scalar))

	def check_backward(self, vector, scalar, y_grad):
		args = (vector, scalar)
		gradient_check.check_backward(vae_m2.Multiply(), args, y_grad, eps=1e-2)

	@condition.retry(3)
	def test_backward_cpu(self):
		self.check_backward(self.vector, self.scalar, self.gy)

	@attr.gpu
	@condition.retry(3)
	def test_backward_gpu(self):
		self.check_backward(cuda.to_gpu(self.vector), cuda.to_gpu(self.scalar), cuda.to_gpu(self.gy))

testing.run_module(__name__, __file__)