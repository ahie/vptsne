import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform.resource_loader import get_data_files_path
from tensorflow.python.framework.load_library import load_op_library
from tensorflow.python.framework import ops
from tensorflow.python import debug as tf_debug
from .neural_network import NeuralNetwork

data_files_path = tf.resource_loader.get_data_files_path()
_tsne_op_module = tf.load_op_library(os.path.join(data_files_path, "tsne_loss.so"))

tsne_loss = _tsne_op_module.tsne_loss

@ops.RegisterGradient("TsneLoss")
def tsne_loss_grad(op, *grad):
  """
  The tf op to compute the gradient of the `tsne_loss` op.
  """
  return [None, grad[0] * _tsne_op_module.tsne_loss_grad(
    op.inputs[0], op.inputs[1],
    op.outputs[1], op.outputs[2], op.outputs[3])]

def example_network_builder(x, **kwargs):
  """
  An example neural network builder function for use with `PTSNE` and `VPTSNE`.
  """
  def _bn(x):
    return tf.layers.batch_normalization(x, training=kwargs["training"])

  layer_0 = _bn(x)
  layer_1 = _bn(tf.layers.dense(layer_0, 250, activation=tf.nn.relu))
  layer_2 = _bn(tf.layers.dense(layer_1, 2500, activation=tf.nn.relu))

  return tf.layers.dense(layer_2, 2, activation=None)

class PTSNE(NeuralNetwork):
  """
  Parametric t-distributed Stochastic Neighbor Embedding [1, 2].

  Parameters
  ----------
  input_shape : array
    Shape of the input data.

  network_builder : function
    A function that takes as a parameter the input placeholder tensor
    and returns a tensor representing the final embedded points.

  learning_rate : Tensor or float, optional (default: 0.001)

  perplexity : float, optional (default: 15)

  exact : bool, optional (default: True)

  delta : float, optional (default: 0.001)

  rho : float, optional (default: 0.5)

  n_neighbors : int, optional (default: 45)

  n_max_candidates : int, optional (default: 60)

  n_max_iters : int, optional (default: 10)

  weights : str, optional (default: None)

  debug : bool, optional (default: False)

  References
  ----------

  [1] L.J.P. van der Maaten and G.E. Hinton. Visualizing data using t-SNE.
      Journal of Machine Learning Research, 9(Nov):2431–2456, 2008.

  [2] L.J.P. van der Maaten. Learning a parametric embedding by preserving local structure.
      In Proceedings of the International Conference on Artificial Intelligence and Statistics, JMLR W&CP, volume 5, pages 384–391, 2009.
  """
  def __init__(
    self,
    input_shape,
    network_builder,
    learning_rate=0.001,
    perplexity=30,
    exact=True,
    delta=0.001,
    rho=0.5,
    n_neighbors=45,
    n_max_candidates=60,
    n_max_iters=10,
    weights=None,
    debug=False):
    
    self.input_shape = input_shape
    self.network_builder = network_builder
    self.perplexity = perplexity
    self.exact = exact
    self.delta = delta
    self.rho = rho
    self.n_neighbors = n_neighbors
    self.n_max_candidates = n_max_candidates
    self.n_max_iters = n_max_iters

    super().__init__(learning_rate, weights, debug)

  def _init_network(self):
    self.x = tf.placeholder(dtype=tf.float32, shape=[None, *self.input_shape], name="x")
    self.y = self.network_builder(self, self.x)
    self.loss = tsne_loss(
      self.x, self.y,
      perplexity=self.perplexity,
      exact=self.exact,
      delta=self.delta,
      rho=self.rho,
      n_neighbors=self.n_neighbors,
      n_max_candidates=self.n_max_candidates,
      n_max_iters=self.n_max_iters).loss

  def fit(self, X=None, y=None, **fit_params):
    """
    Fit X into an embedded space.

    Parameters
    ----------
    X : array_like, shape (n_samples, *input_shape)

    y : Ignored

    Returns
    -------
    self
    """

    if "weights" in fit_params:
      self.saver.restore(self.session, fit_params["weights"])

    self._fit(X, **fit_params)

    return self

  def transform(self, X, y=None):
    """
    Apply dimensionality reduction on X.

    Parameters
    ----------
    X : array_like, shape (n_samples, *input_shape)

    y : Ignored

    Returns
    -------
    X_new : array_like, shape (n_samples, n_output_dimensions)
        Embedding of the given X in low-dimensional space.
    """
    return self.session.run(
      self.y,
      feed_dict={self.x: X})

  def fit_transform(self, X, y=None, **fit_params):
    """
    Fit X into an embedded space and return that transformed
    Output.

    Parameters
    ----------
    X : array_like, shape (n_samples, *input_shape)

    y : Ignored

    Returns
    -------
    X_new : array_like, shape (n_samples, n_output_dimensions)
        Embedding of the training data in low-dimensional space.
    """
    self.fit(X, None, **fit_params)
    return self.transform(X)

  def _fit(self, X, n_iters=1500, batch_size=200, **fit_params):
    """
    Internal implementation for fit.
    """

    if X is None:
      raise ValueError("cannot call fit on PTSNE with no data (X == None)")

    for i in range(n_iters):
      idx = np.random.choice(X.shape[0], batch_size, replace=False)
      _, loss = self.session.run(
        [self.train_op, self.loss],
        feed_dict={
          self.x: X[idx, :],
          self.training: True})

      if "hook_fn" in fit_params:
        fit_params["hook_fn"]([self, i, loss])

class VPTSNE(PTSNE):
  """
  Variational Parametric t-distributed Stochastic Neighbor Embedding.

  Parameters
  ----------
  vae : VAE, optional (default: None)
    Variational autoencoder to use in training the embedding.

  network_builder : function
    A function that takes as a parameter the input placeholder tensor
    and returns a tensor representing the final embedded points.

  learning_rate : Tensor or float, optional (default: 0.001)

  perplexity : float, optional (default: 15)

  exact : bool, optional (default: True)

  delta : float, optional (default: 0.001)

  rho : float, optional (default: 0.5)

  n_neighbors : int, optional (default: 45)

  n_max_candidates : int, optional (default: 60)

  n_max_iters : int, optional (default: 10)

  weights : str, optional (default: None)

  debug : bool, optional (default: False)
  """
  def __init__(
    self,
    vae,
    network_builder,
    learning_rate=0.001,
    perplexity=30,
    exact=True,
    delta=0.001,
    rho=0.5,
    n_neighbors=45,
    n_max_candidates=60,
    n_max_iters=10,
    weights=None,
    debug=False):

    super().__init__(
      vae.output_shape,
      network_builder,
      learning_rate,
      perplexity,
      exact,
      delta,
      rho,
      n_neighbors,
      n_max_candidates,
      n_max_iters,
      weights,
      debug)

    self.vae = vae

  def transform(self, X, y=None, reconstruct=True, deterministic=True):
    """
    Apply dimensionality reduction on X.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_input_dimensions)

    y : Ignored

    reconstruct : bool, optional (default: True)
      Whether to run the samples X through the underlying VAE
      before transforming.

    deterministic : bool, optional (default: True)
      Whether to run the reconstruction deterministically.

    Returns
    -------
    X_new : array_like, shape (n_samples, n_output_dimensions)
        Embedding of the given X in low-dimensional space.
    """
    return self.session.run(
      self.y,
      feed_dict={
        self.x: self.vae.reconstruct(X, deterministic=deterministic) if reconstruct else X})

  def fit(self, X=None, y=None, **fit_params):
    """
    Learn a parametric embedding from the decoder outputs of the
    VAE this instance was constructed with.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_input_dimensions)
      Data to reconstruct through the VAE and fit the embedding to.
      If None, the training data will be generated from the VAE by reconstructing
      samples from its prior distribution.

    y : Ignored

    Returns
    -------
    self
    """
    return super().fit(X, y, **fit_params)

  def _fit(
    self,
    X=None,
    n_iters=1500,
    batch_size=200,
    deterministic=True,
    fit_vae=True,
    n_vae_iters=10000,
    vae_batch_size=None,
    vae_weights=None,
    **fit_params):
    """
    Internal implementation for fit.
    """

    if fit_vae:
      self.vae.fit(X, None, n_vae_iters, vae_batch_size, vae_weights, **fit_params)

    for i in range(n_iters):

      if X is not None:
        indices = np.random.choice(X.shape[0], batch_size, replace=False)
        minibatch = self.vae.reconstruct(X[indices, :], deterministic=deterministic)
      else:
        minibatch = self.vae.sample_prior(batch_size)

      _, loss = self.session.run(
        [self.train_op, self.loss],
        feed_dict={
          self.x: minibatch,
          self.training: True})

      if "hook_fn" in fit_params:
        fit_params["hook_fn"]([self, i, loss])

  def save_weights(self, file_name, vae_file_name=None):
    """
    Save the weights of this neural network to the specified file.

    Parameters
    ----------
    file_name : str

    vae_file_name : str, optional (default: None)
    """
    if vae_file_name is not None:
      self.vae.save_weights(vae_file_name)
    super().save_weights(file_name)

  def load_weights(self, file_name, vae_file_name=None):
    """
    Load the weights of this neural network from the specified file.

    Parameters
    ----------
    file_name : str

    vae_file_name : str, optional (default: None)
    """
    if vae_file_name is not None:
      self.vae.load_weights(vae_file_name)
    super().load_weights(file_name)

