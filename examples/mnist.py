import numpy as np
import tensorflow as tf
import tensorflow.distributions as tfds
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from vptsne import (VAE, VPTSNE)
from vptsne.helpers import *
from sklearn.manifold.t_sne import trustworthiness

def display_reconstructions(n_plots):
  for i in range(n_plots):
    plt.subplot(211)
    plt.imshow(mnist.train._images[i].reshape((28, 28)))
    reconstructed = vae.reconstruct([mnist.train._images[i]])[0].reshape((28, 28))
    plt.subplot(212)
    plt.imshow(reconstructed)
    plt.show()

def distances(x):
  tiled = np.tile(np.expand_dims(x, 1), np.stack([1, x.shape[0], 1]))
  diff = tiled - np.transpose(tiled, [1, 0, 2])
  return np.sum(np.square(diff), axis=2)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

print(np.var(distances(mnist.test._images[:1000])))

n_input_dimensions = mnist.train._images.shape[1]
n_latent_dimensions = 3

vae_layer_definitions = [
  (256, tf.nn.relu),
  (128, tf.nn.relu),
  (32, tf.nn.relu)]

vae_encoder_layers = LayerDefinition.from_array(vae_layer_definitions)
vae_decoder_layers = LayerDefinition.from_array(reversed(vae_layer_definitions))

vae = VAE(
  [n_input_dimensions],
  get_gaussian_network_builder(vae_encoder_layers, n_latent_dimensions),
  gaussian_prior_supplier,
  gaussian_supplier,
  get_bernoulli_network_builder(vae_decoder_layers, n_input_dimensions),
  bernoulli_supplier)

vae.fit(mnist.train._images, n_iters=14000, batch_size=1000, hook_fn=print)

from sklearn.neighbors import KNeighborsClassifier as KNC
print(
  "1-NN, test set",
  KNC(n_neighbors=1)
  .fit(vae.transform(mnist.train._images), mnist.train._labels)
  .score(vae.transform(mnist.test._images), mnist.test._labels))

def tt():
  
  vptsne_layers = LayerDefinition.from_array([
    (200, tf.nn.relu),
    (200, tf.nn.relu),
    (2000, tf.nn.relu),
    (2, None)])
  
  from vptsne import PTSNE
  vptsne = VPTSNE(
#    [n_input_dimensions],
    vae,
    get_feed_forward_network_builder(vptsne_layers), perplexity=30)
  
  fit_params = {
    "hook_fn": print,
    "n_iters": 2000,
    "batch_size": 200,
    "deterministic": False,
    "fit_vae": False,
    "n_vae_iters": 14000,
    "vae_batch_size": 1000}
  
  #vptsne.load_weights("models/mnist_vptsne.ckpt", "models/mnist_vae.ckpt")
  vptsne.fit(mnist.train._images, **fit_params)
  #vptsne.save_weights("models/mnist_vptsne.ckpt", "models/mnist_vae.ckpt")
  #display_reconstructions(10)
  
  #from sklearn.decomposition import PCA
  #p = PCA(n_components=3).fit(mnist.train._images)
  #train = p.transform(mnist.train._images)
  #vptsne.fit(train, **fit_params)
  
  #transformed = vptsne.transform(train)
  transformed = vptsne.transform(mnist.train._images, reconstruct=True)
  #transformed = vae.transform(mnist.train._images)
  
  transformed_test = vptsne.transform(mnist.test._images, reconstruct=True)
  
  print(
    "Trustworthiness, test set",
    trustworthiness(
      mnist.test._images,
      transformed_test,
      n_neighbors=12))
  
  #print(
  #  "Trustworthiness, first 10k",
  #  trustworthiness(
  #    mnist.train._images[:10000],
  #    vptsne.transform(mnist.train._images[:10000]),
  #    n_neighbors=12))
  
  from sklearn.neighbors import KNeighborsClassifier as KNC
  print(
    "1-NN, test set",
    KNC(n_neighbors=1)
    .fit(transformed, mnist.train._labels)
    .score(transformed_test, mnist.test._labels))

  plt.clf()
  color_palette = np.random.rand(100, 3)
  for label in np.unique(mnist.train._labels):
    tmp = transformed[mnist.train._labels == label]
    plt.scatter(tmp[:, 0], tmp[:, 1], s=0.2, c=color_palette[label])
  plt.show()

for i in range(10):
  tt()

#color_palette = np.random.rand(100, 3)
#for label in np.unique(mnist.train._labels):
#  tmp = transformed[mnist.train._labels == label]
#  plt.scatter(tmp[:, 0], tmp[:, 1], s=0.2, c=color_palette[label])
#plt.show()

