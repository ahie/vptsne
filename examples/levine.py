import numpy as np
import tensorflow as tf
import tensorflow.distributions as tfds
import matplotlib.pyplot as plt
from vptsne import (VAE, VPTSNE)
from vptsne.helpers import *
from sklearn.manifold.t_sne import trustworthiness

levine_tsv = np.loadtxt("CYTOMETRY_data/levine.tsv", delimiter="\t", skiprows=1)
levine_data = levine_tsv[:,:levine_tsv.shape[1] - 1]
levine_labels = levine_tsv[:,levine_tsv.shape[1] - 1].astype(int)

n_input_dimensions = levine_data.shape[1]
n_latent_dimensions = 2

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
  get_gaussian_network_builder(vae_decoder_layers, n_input_dimensions, constant_sigma=0.025),
  gaussian_supplier)

vptsne_layers = LayerDefinition.from_array([
  (250, tf.nn.relu),
  (2500, tf.nn.relu),
  (2, None)])

vptsne = VPTSNE(
  vae,
  get_feed_forward_network_builder(vptsne_layers),
  perplexity=7., learning_rate=0.01)

fit_params = {
  "n_iters": 1000,
  "batch_size": 300,
  "deterministic": False,
  "fit_vae": True,
  "n_vae_epochs": 300,
  "vae_batch_size": 10000,
  "hook_fn": print}

#vptsne.load_weights("models/levine_vptsne.ckpt", "models/levine_vae.ckpt")
vptsne.fit(levine_data, **fit_params)
vptsne.save_weights("models/levine_vptsne.ckpt", "models/levine_vae.ckpt")

transformed = vptsne.transform(levine_data)

print(
  "Trustworthiness, first 10k subset",
  trustworthiness(
    levine_data[:10000],
    transformed[:10000],
    n_neighbors=12))

color_palette = np.random.rand(100, 3)
for label in np.unique(levine_labels):
  tmp = transformed[levine_labels == label]
  plt.scatter(tmp[:, 0], tmp[:, 1], s=0.2, c=color_palette[label])
plt.show()

