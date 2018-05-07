import tensorflow as tf
import tensorflow.contrib.distributions as tfds
from .bijectors import (PlanarFlow, InverseAutoregressiveFlow)

class LayerDefinition(object):
  def __init__(self, shape, activation):
    self.shape = shape
    self.activation = activation

  @staticmethod
  def from_array(layer_definitions_as_array):
    return [
      LayerDefinition(layer_definition[0], layer_definition[1])
      for layer_definition in layer_definitions_as_array]

def get_feed_forward_network_builder(layer_definitions, **kwargs):

  if "batch_normalization" in kwargs:
    apply_bn = kwargs["batch_normalization"]
  else:
    apply_bn = True

  def _bn(x, apply_bn, training):
    if not apply_bn:
      return x
    return tf.layers.batch_normalization(x, training=training)

  def _builder(neural_network, x, **builder_kwargs):

    training = neural_network.training
    layer = x

    for layer_definition in layer_definitions:
      layer = tf.layers.dense(
        _bn(layer, apply_bn, training),
        layer_definition.shape,
        layer_definition.activation)

    return layer

  return _builder

def get_gaussian_network_builder(layer_definitions, n_output_dimensions, constant_sigma=None, output_hidden=False, **kwargs):

  def _builder(neural_network, x, **builder_kwargs):
    layer = get_feed_forward_network_builder(layer_definitions, **kwargs)(neural_network, x, **builder_kwargs)
    mu = tf.layers.dense(layer, n_output_dimensions, activation=None)
    return {
      "output": x if output_hidden else mu,
      "mu": mu,
      "log_sigma_sq": tf.fill(tf.shape(mu), tf.log(tf.square(constant_sigma)))
        if constant_sigma is not None else tf.layers.dense(layer, n_output_dimensions, activation=None),
      "flow_input": tf.layers.dense(layer, n_output_dimensions, activation=None)}

  return _builder

def get_bernoulli_network_builder(layer_definitions, n_output_dimensions, output_hidden=False, **kwargs):

  def _builder(neural_network, x, **builder_kwargs):
    layer = get_feed_forward_network_builder(layer_definitions, **kwargs)(neural_network, x, **builder_kwargs)
    probs = tf.layers.dense(layer, n_output_dimensions, activation=tf.nn.sigmoid)
    return {
      "output": x if output_hidden else probs,
      "probs": probs}

  return _builder

def gaussian_prior_supplier(vae):
  return tfds.MultivariateNormalDiag(
    loc=tf.zeros(vae.n_latent_dimensions, tf.float32))

def gaussian_supplier(vae, network_output):
  return tfds.MultivariateNormalDiag(
    loc=network_output["mu"],
    scale_diag=tf.exp(0.5 * network_output["log_sigma_sq"]) + 1e-5) # +1e-5 fudge factor to avoid NaN if sigma -> 0

def bernoulli_supplier(vae, network_output):
  return tfds.Bernoulli(probs=network_output["probs"])

def d_tanh(x):
  with tf.name_scope("d_tanh"):
    return 1. - tf.square(tf.tanh(x))

def get_planar_flow_supplier(n_flows, non_linearity=tf.tanh, d_non_linearity=d_tanh):

  def _supplier(vae):
    flows = [PlanarFlow(vae.encoder_output["flow_input"], vae.sample, non_linearity, d_non_linearity) for _ in range(n_flows)]
    return tfds.bijectors.Chain(flows)

  return _supplier

def get_iaf_supplier(n_flows):

  def _supplier(vae):
    flows = [InverseAutoregressiveFlow() for _ in range(n_flows)]
    return tfds.bijectors.Chain(flows)

  return _supplier

