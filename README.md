## Semi-Supervised Learning with Deep Generative Models

Chainer implementation of Variational AutoEncoder(VAE) model M1, M2, M1+M2 

### Requirements

- Chainer 1.8+
- sklearn

To visualize results, you need

- matplotlib.patches
- PIL
- pandas

### M1

#### Training details

| params | value |
|:-----------|------------:|
| OS | Windows |
| GPU | GeForce GTX 970M |
| ndim_z | 2 |
| encoder_apply_dropout | False |
| decoder_apply_dropout | False |
| encoder_apply_batchnorm | True |
| decoder_apply_batchnorm | True |
| encoder_apply_batchnorm_to_input | True |
| decoder_apply_batchnorm_to_input | True |
| encoder_units | [600, 600] |
| decoder_units | [600, 600] |

#### Result

##### Latent space

### M2

#### Training details

| params | value |
|:-----------|------------:|
| OS | Windows |
| GPU | GeForce GTX 970M |
| ndim_z | 50 |
| encoder_xy_z_apply_dropout | False |
| encoder_x_y_apply_dropout | False |
| decoder_apply_dropout | False |
| encoder_xy_z_apply_batchnorm_to_input | True |
| encoder_x_y_apply_batchnorm_to_input | True |
| decoder_apply_batchnorm_to_input | True |
| encoder_xy_z_apply_batchnorm | True |
| encoder_x_y_apply_batchnorm | True |
| decoder_apply_batchnorm | True |
| encoder_xy_z_hidden_units | [500] |
| encoder_x_y_hidden_units | [500] |
| decoder_hidden_units | [500] |
| batchnorm_before_activation | True |

#### Result

##### Classification

| data | # |
|:-----------|------------:|
| labeled | 100 |
| unlabeled | 49900 |
| validation | 10000 |
| test | 10000 |

| * | # |
|:-----------|------------:|
| epochs | 490 |
| minutes | 1412 |
| params updates per an epoch | 2000 |

Validation accuracy:

![M2](http://musyoku.github.io/images/post/2016-07-02/m2_validation_accuracy.png)

Test accuracy: **0.9018**


##### Analogies

| data | # |
|:-----------|------------:|
| labeled | 10000 |
| unlabeled | 40000 |
