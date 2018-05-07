import tensorflow as tf
from tensorflow.python import debug as tf_debug
from abc import (ABCMeta, abstractmethod)

class NeuralNetwork(metaclass=ABCMeta):
  """
  Abstract base class for neaural networks.

  Parameters
  ----------
  learning_rate : Tensor or float, optional (default: 0.001)

  weights : str, optional (default: None)

  debug : bool, optional (default: False)
  """
  def __init__(
    self,
    learning_rate=0.001,
    weights=None,
    debug=False):

    self.learning_rate = learning_rate
    self.graph = tf.Graph() 

    with self.graph.as_default():
      self.training = tf.placeholder_with_default(False, shape=(), name="training")

      self._init_network()

      with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

      self.session = tf.Session()
      if debug:
        self.session = tf_debug.LocalCLIDebugWrapperSession(self.session)

      self.saver = tf.train.Saver()
      if weights:
        self.load_weights(weights)
      else:
        self.reinitialize_weights()

  def save_weights(self, file_name):
    """
    Save the weights of this neural network to the specified file.

    Parameters
    ----------
    file_name : str
    """
    self.saver.save(self.session, file_name)

  def load_weights(self, file_name):
    """
    Load the weights of this neural network from the specified file.

    Parameters
    ----------
    file_name : str
    """
    self.saver.restore(self.session, file_name)

  def reinitialize_weights(self, random_seed=None):
    """
    Reinitialize all variables of this model.

    Parameters
    ----------
    random_seed : int, optional (default: none)
    """
    with self.graph.as_default():
      if random_seed is not None:
        tf.set_random_seed(random_seed)
      self.session.run(tf.global_variables_initializer())
      self.session.run(tf.local_variables_initializer())

  def set_learning_rate(learning_rate):
    """
    Set the learning rate of this network.

    Parameters
    ----------
    learning_rate : Tensor or float, optional (default: 0.001)
    """
    with self.graph.as_default():
      with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

  @abstractmethod
  def _init_network(self):
    """
    Internal method for defining the computation graph. Must be overridden by subclasses.
    """
    pass

