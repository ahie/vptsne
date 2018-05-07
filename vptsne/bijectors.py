import tensorflow as tf
import tensorflow.contrib.distributions as tfds

class PlanarFlow(tfds.bijectors.Bijector):
  """
  Planar flow [1] as a TensorFlow Bijector.

  Parameters
  ----------

  References
  ----------

  [1] Rezende, D. and Mohamed, S. (2015).
      Variational inference with normalizing flows.
      In Proceedings of The 32nd International Conference on Machine Learning, pages 1530–1538.
  """
  def __init__(
    self,
    x,
    z,
    non_linearity,
    d_non_linearity,
    forward_min_event_ndims=2,
    validate_args=False,
    name="planar_flow"):

    super().__init__(
      forward_min_event_ndims=forward_min_event_ndims,
      validate_args=validate_args,
      name=name)

    self.non_linearity = non_linearity
    self.d_non_linearity = d_non_linearity

    self.u = tf.layers.dense(x, z.shape[1], activation=None)
    self.w = tf.layers.dense(x, z.shape[1], activation=None)
    self.b = tf.layers.dense(x, 1, activation=None)

    def _u_hat(u, w):
      wtu = tf.reduce_sum(w * u, axis=1, keep_dims=True)
      return u + (-1. + tf.maximum(0., wtu) - wtu) * (w / tf.square(tf.norm(w, axis=1, keep_dims=True)))

    # Enforce invertibility
    self.u = _u_hat(self.u, self.w)

  def _forward(self, x):
    return x + self.u * self.non_linearity(tf.reduce_sum(self.w * x, axis=1, keep_dims=True) + self.b)

  def _inverse(self, y):
    # directly uses the caching mechanism of Bijector
    return self._call_inverse(y, "inverse")

  def _inverse_log_det_jacobian(self, y):
    x = self._inverse(y)
    psi = self.d_non_linearity(tf.reduce_sum(self.w * x, axis=1, keep_dims=True) + self.b) * self.w
    return tf.log(tf.abs(1. + tf.reduce_sum(self.u * psi, axis=1)) + 1e-20)

  def _forward_log_det_jacobian(self, x):
    raise NotImplementedError

class InverseAutoregressiveFlow(tfds.bijectors.Bijector):
  """
  Inverse autoregressive flow (IAF) [1] as a TensorFlow Bijector.

  Parameters
  ----------

  References
  ----------

  [1] Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling.
      Improving Variational Inference with Inverse Autoregressive Flow.
      In Neural Information Processing Systems, 2016.
  """
  def __init__(
    self,
    hidden_layers=[256, 128],
    forward_min_event_ndims=2,
    validate_args=False,
    name="inverse_autoregressive_flow"):

    super().__init__(
      forward_min_event_ndims=forward_min_event_ndims,
      validate_args=validate_args,
      name=name)

    maf_template = tfds.bijectors.masked_autoregressive_default_template
    maf = tfds.bijectors.MaskedAutoregressiveFlow(
      shift_and_log_scale_fn=maf_template(hidden_layers=hidden_layers))
    self.iaf = tfds.bijectors.Invert(maf)

  def _forward(self, x):
    return self.iaf._forward(x)

  def _inverse(self, y):
    return self.iaf._inverse(y)

  def _inverse_log_det_jacobian(self, y):
    return self.iaf._inverse_log_det_jacobian(y)

  def _forward_log_det_jacobian(self, x):
    return self.iaf._forward_log_det_jacobian(x)

