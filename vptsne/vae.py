import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as tfds
from .neural_network import NeuralNetwork

class VAE(NeuralNetwork):
  """
  Variational autoencoder [1, 2], with optional normalizing flows [3].

  References
  ----------

  [1] Kingma, D. P. and Welling, M. Auto-encoding variational Bayes. In ICLR, 2014.

  [2] Rezende, D. J., Mohamed, S., and Wierstra, D. (2014).
      Stochastic backpropagation and approximate inference in deep generative models.
      In Proceedings of the 31st International Conference on Machine Learning (ICML-14), pages 1278–1286.

  [3] Rezende, D. and Mohamed, S. (2015).
      Variational inference with normalizing flows.
      In Proceedings of The 32nd International Conference on Machine Learning, pages 1530–1538.
  """
  def __init__(
    self,
    input_shape,
    encoder_network_builder,
    prior_distribution_supplier,
    posterior_distribution_supplier,
    decoder_network_builder,
    output_distribution_supplier,
    bijector_supplier=None,
    learning_rate=0.0005,
    weights=None,
    debug=False):

    self.input_shape = input_shape
    self.encoder_network_builder = encoder_network_builder
    self.prior_distribution_supplier = prior_distribution_supplier
    self.posterior_distribution_supplier = posterior_distribution_supplier
    self.bijector_supplier = bijector_supplier
    self.decoder_network_builder = decoder_network_builder
    self.output_distribution_supplier = output_distribution_supplier

    super().__init__(learning_rate, weights, debug)

  def _init_network(self):
    """
    Internal method for defining the computation graph.
    """

    self.x = tf.placeholder(dtype=tf.float32, shape=[None, *self.input_shape], name="x")
    self.deterministic = tf.placeholder_with_default(False, shape=(), name="deterministic")

    self.encoder_output = self.encoder_network_builder(self, self.x)
    self.posterior_distribution = self.posterior_distribution_supplier(self, self.encoder_output)
    self.posterior_distribution_mean = self.posterior_distribution.mean()

    if self.posterior_distribution.reparameterization_type is not tfds.FULLY_REPARAMETERIZED:
      raise Exception("the supplied posterior distribution is not reparameterizable")

    self.posterior_distribution_sample = self.posterior_distribution.sample()

    self.sample = tf.cond(
      self.deterministic,
      lambda: self.posterior_distribution_mean,
      lambda: self.posterior_distribution_sample)

    if self.bijector_supplier is not None:
      self.bijector = self.bijector_supplier(self)
      self.transformed_sample = self.bijector.forward(self.sample)
      self.sildj = tf.reduce_mean(self.bijector._inverse_log_det_jacobian(self.transformed_sample))
    else:
      self.transformed_sample = self.sample
      self.sildj = 0.

    self.n_latent_dimensions = self.transformed_sample.shape[1]
    self.prior_distribution = self.prior_distribution_supplier(self)

    self.decoder_output = self.decoder_network_builder(self, self.transformed_sample)
    self.output_distribution = self.output_distribution_supplier(self, self.decoder_output)

    if "output" not in self.decoder_output:
      raise Exception("decoder_network_builder must return a dict containing 'output'")
    self.output = self.decoder_output["output"]
    self.output_shape = self.output.shape[1:]

    self.lpx = self.output_distribution.log_prob(self.x)
    self.lpx = tf.reduce_sum(self.lpx, axis=tf.range(1, tf.rank(self.lpx)))

    self.lqz0 = self.posterior_distribution.log_prob(self.sample)
    self.lpzk = self.prior_distribution.log_prob(self.transformed_sample)

    self.loss = -tf.reduce_mean(self.lpx) - self.sildj + tf.reduce_mean(self.lqz0) - tf.reduce_mean(self.lpzk)

    self.marginal = tf.reduce_logsumexp(self.lpx + self.lpzk - self.lqz0) - tf.log(tf.cast(tf.shape(self.x)[0], self.x.dtype))

  def reconstruct(self, X, deterministic=False):
    """
    Reconstruct the given data with this VAE.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_input_dimensions)
      Data to reconstruct through this VAE.

    deterministic : bool, optional (default: False)

    Returns
    -------
    X_reconstructed : array_like, shape (n_samples, n_output_dimensions)
      The reconstruction of X produced by this VAE.
    """
    return self.session.run(
      self.output,
      feed_dict={
        self.x: X,
        self.deterministic: deterministic})

  def sample_prior(self, n_samples):

    samples = self.session.run(
      self.prior_distribution.sample(n_samples))

    return self.session.run(
      self.output,
      feed_dict={
        self.sample: samples})

  def fit(self, X, y=None, n_iters=10000, batch_size=100, weights=None, **fit_params):

    if weights is not None:
      self.saver.restore(self.session, weights)

    batch_size = int(X.shape[0] * 0.1) if batch_size is None else batch_size

    for i in range(n_iters):
      indices = np.random.choice(X.shape[0], batch_size, replace=False)
      batch = X[indices]
      _, loss = self.session.run(
        [self.train_op, self.loss],
        feed_dict={
          self.x: batch,
          self.training: True})

      if "hook_fn" in fit_params:
        fit_params["hook_fn"]([self, i, loss])

  def transform(self, X, y=None):
    return self.session.run(
      self.posterior_distribution_mean,
      feed_dict={self.x: X})

  def fit_transform(self, X, y=None, **fit_params):
    self.fit(X, None, **fit_params)
    return self.transform(X)

  def score(self, X, n_samples=1000):
    """
    Compute the approximate log-marginal (log p(x)) for each data point in X under this VAE.

    X : array_like, shape (n_samples, *input_shape)

    n_samples: int, optional (default: 1000)

    Returns
    -------
    P_x : array_like, shape (n_samples)
      The estimated marginal for each data point in X.
    """
    def score_sample(x):
      return self.session.run(
        self.marginal,
        feed_dict={self.x: [x] * n_samples})
    return np.apply_along_axis(score_sample, 1, X)

