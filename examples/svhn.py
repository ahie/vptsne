import os
import numpy as np
import tensorflow as tf
import tensorflow.distributions as tfds
import matplotlib.pyplot as plt
import scipy.io as sio
from vptsne import (VAE, VPTSNE)
from vptsne.helpers import *

def display_reconstructions(n_plots):
  for i in range(n_plots):
    plt.subplot(211)
    plt.imshow(svhn_train[i])
    reconstructed = vae.reconstruct([svhn_train[i]])[0]
    plt.subplot(212)
    plt.imshow(reconstructed)
    plt.show()

svhn_train_file_path = "SVHN_data/train_32x32.mat"
svhn_extra_file_path = "SVHN_data/extra_32x32.mat"
svhn_test_file_path = "SVHN_data/test_32x32.mat"

if not os.path.exists(svhn_train_file_path):
  raise FileNotFoundError("SVHN training data file not found, run get_svhn_data.sh")
if not os.path.exists(svhn_extra_file_path):
  raise FileNotFoundError("SVHN extra data file not found, run get_svhn_data.sh")
if not os.path.exists(svhn_test_file_path):
  raise FileNotFoundError("SVHN test data file not found, run get_svhn_data.sh")

def load_and_transform_svhn_file(svhn_file_path):
  svhn = sio.loadmat(svhn_file_path)["X"]
  svhn = np.transpose(svhn, [3, 0, 1, 2])
  svhn = svhn / float(255)
  return svhn

svhn_train = load_and_transform_svhn_file(svhn_train_file_path)
svhn_extra = load_and_transform_svhn_file(svhn_extra_file_path)
svhn_test = load_and_transform_svhn_file(svhn_test_file_path)

input_shape = svhn_train.shape[1:]
n_latent_dimensions = 100

encoder_hidden_3_shape = None
encoder_flattened_shape = None

def relu_bn(x, training):
  return tf.nn.relu(tf.layers.batch_normalization(x, training=training))

def encoder_network_builder(x, training):
  hidden_1 = relu_bn(tf.layers.conv2d(x, 128, 2, [2, 2]), training)
  hidden_2 = relu_bn(tf.layers.conv2d(hidden_1, 128 * 2, 2, [2, 2]), training)
  hidden_3 = relu_bn(tf.layers.conv2d(hidden_2, 128 * 4, 2, [2, 2]), training)
  flattened = tf.layers.Flatten()(hidden_3)

  global encoder_hidden_3_shape, encoder_flattened_shape
  encoder_hidden_3_shape = hidden_3.shape[1:]
  encoder_flattened_shape = flattened.shape[1]

  return {
    "mu": tf.layers.dense(flattened, n_latent_dimensions, activation=None),
    "log_sigma_sq": tf.layers.dense(flattened, n_latent_dimensions, activation=None)}

def decoder_network_builder(z, training):
  hidden_0 = tf.layers.dense(z, encoder_flattened_shape, activation=tf.nn.relu)
  hidden_1 = tf.reshape(hidden_0, [-1, *encoder_hidden_3_shape])
  hidden_2 = relu_bn(tf.layers.conv2d_transpose(hidden_1, 128 * 4, 2, [2, 2]), training)
  hidden_3 = relu_bn(tf.layers.conv2d_transpose(hidden_2, 128 * 2, 2, [2, 2]), training)
  hidden_4 = relu_bn(tf.layers.conv2d_transpose(hidden_3, 128, 2, [2, 2]), training)
  hidden_5 = tf.nn.sigmoid(tf.layers.conv2d(hidden_4, 3, 1, [1, 1]))
  return {
    "output": z,
    "probs": hidden_5}

vae = VAE(
  input_shape,
  encoder_network_builder,
  gaussian_prior_supplier,
  gaussian_supplier,
  decoder_network_builder,
  bernoulli_supplier,
  learning_rate=0.00001)

#vae.load_weights("models/svhn_vae.ckpt")
vae.fit(svhn_train, n_epochs=200, batch_size=1000, hook_fn=print)
vae.save_weights("models/svhn_vae.ckpt")
#display_reconstructions(10)

vptsne_layers = LayerDefinition.from_array([
  (250, tf.nn.relu),
  (2500, tf.nn.relu),
  (2, None)])

vptsne = VPTSNE(
  vae,
  get_feed_forward_network_builder(vptsne_layers), learning_rate=0.00001, perplexity=10.)

fit_params = {
  "n_iters": 1500,
  "batch_size": 500,
  "fit_vae": False,
  "hook_fn": print}

#vptsne.load_weights("models/svhn_vptsne.ckpt")
vptsne.fit(svhn_train, **fit_params)
vptsne.save_weights("models/svhn_vptsne.ckpt")

transformed = vptsne.transform(svhn_test)

plt.scatter(transformed[:, 0], transformed[:, 1], s=0.1)
plt.show()

