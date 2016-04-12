# -*- coding: utf-8 -*-
from activations import activations

class Config:
	def __init__(self):
		# 共通設定
		self.img_channel = 1
		self.img_width = 28

		## Batch Normalizationを使うかどうか
		self.apply_batchnorm = True

		## GPUを使うかどうか
		self.use_gpu = True

		## 入力ベクトルの次元
		self.n_x = self.img_width ** 2

		## 隠れ変数ベクトルの次元
		self.n_z = 100

		## Default: 1.0
		## 重みの初期化
		self.wscale = 0.1

		# Encoderの設定
		## 隠れ層のユニット数
		## 左から入力層側->出力層側に向かって各層のユニット数を指定
		self.enc_n_hidden_units = [600, 600]

		## 活性化関数
		## See activations.py
		self.enc_activation_type = "elu"

		## 出力層の活性化関数
		## Noneも可
		self.enc_output_activation_type = None

		## 出力層でBatch Normalizationを使うかどうか
		self.enc_apply_batchnorm_to_output = False

		## ドロップアウト
		self.enc_apply_dropout = True

		# Decoderの設定
		## zからxを復号する
		## 左から入力層側->出力層側に向かって各層のユニット数を指定
		self.dec_n_hidden_units = [600, 600]

		## 隠れ層の活性化関数
		## See activations.py
		self.dec_activation_type = "elu"

		## 出力層（画素値を表す）の活性化関数
		## 通常入力画像の画素値の範囲は-1から1に正規化されているためtanhを使う
		## Noneも可
		self.dec_output_activation_type = "tanh"

		## 入力層でBatch Normalizationを使うかどうか
		self.dec_apply_batchnorm_to_input = False

		## 出力層でBatch Normalizationを使うかどうか
		self.dec_apply_batchnorm_to_output = False

		## ドロップアウト
		self.dec_apply_dropout = True

		## p(x|z)の種類
		## bernoulli or gaussian
		self.dec_type = "gaussian"

	def check(self):
		if self.gen_activation_type not in activations:
			raise Exception("Invalid type of activation for gen_activation_type.")
		if self.gen_output_activation_type and self.gen_output_activation_type not in activations:
			raise Exception("Invalid type of activation for gen_output_activation_type.")
		if self.dis_activation_type not in activations:
			raise Exception("Invalid type of activation for dis_activation_type.")
		if self.dis_softmax_activation_type and self.dis_softmax_activation_type not in activations:
			raise Exception("Invalid type of activation for dis_softmax_activation_type.")
		if self.dec_activation_type not in activations:
			raise Exception("Invalid type of activation for dec_activation_type.")
		if self.dec_output_activation_type and self.dec_output_activation_type not in activations:
			raise Exception("Invalid type of activation for dec_output_activation_type.")

		if config.gen_encoder_type not in {"deterministic", "gaussian"}:
				raise Exception("Invalid encoder type for gen_encoder_type.")

	def dump(self):
		pass


config = Config()