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
		self.matrix = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
		self.vector = numpy.random.uniform(-1, 1, (3, 1)).astype(numpy.float32)
		self.gy = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
		self.y = self.matrix * self.vector

	def check_forward(self, matrix, vector, y_expect):
		matrix = chainer.Variable(matrix)
		vector = chainer.Variable(vector)
		y = vae_m2.multiply(matrix, vector)
		gradient_check.assert_allclose(y_expect, y.data)

	@condition.retry(3)
	def test_forward_cpu(self):
		self.check_forward(self.matrix, self.vector, self.y)

	@attr.gpu
	@condition.retry(3)
	def test_forward_gpu(self):
		self.check_forward(cuda.to_gpu(self.matrix), cuda.to_gpu(self.vector), cuda.to_gpu(self.matrix * self.vector))

	def check_backward(self, matrix, vector, y_grad):
		args = (matrix, vector)
		gradient_check.check_backward(vae_m2.Multiply(), args, y_grad, eps=1e-2)

	@condition.retry(3)
	def test_backward_cpu(self):
		self.check_backward(self.matrix, self.vector, self.gy)

	@attr.gpu
	@condition.retry(3)
	def test_backward_gpu(self):
		self.check_backward(cuda.to_gpu(self.matrix), cuda.to_gpu(self.vector), cuda.to_gpu(self.gy))

testing.run_module(__name__, __file__)