import numpy as np

class Layer(object):
  '''
  Represents one layer of a multilayer perceptron (MLP).

  Initializes and updates parameter values, weights (w) and bias (b), for 
  aggregation within the layer.

  '''

  def __init__(self, in_dim=1, out_dim=1, activ_fun=None, init_with_normal=True, 
               mean=0, var=0.055, min=-0.1, max=0.1, activated=None):
    '''
    Constructs a new Layer object.

    Args:
     in_dim (int): Dimension of inputs to this layer
     out_dim (int): Dimension of outputs from this layer
     activ_fun (list or tuple): Container of functions from the Activation 
      class to apply to each layer
     init_with_normal (bool): True to initialize parameters by sampling from a 
      normal distribution, false to sample from a uniform distribution
     mean (float): Mean to use if initializing with a normal dist
     var (float): Variance to use if initializing with a normal dist
     min (float): Minimum bound to use if initializing with a uniform dist
     max (float): Maximum bound to use if initializing with a uniform dist
     activated (ndarray): Array to supply in special cases (i.e., input layer)
      to explicitly set the activation values before any data is passed through
      
    '''
    self.activ_fun = activ_fun
    self.w = None # Weights
    self.b = None # Bias
    self.z = None # Aggregations
    self.a = activated # Activations

    # Initialize layer's parameters if a non-input layer
    if activ_fun is not None:
      if init_with_normal:
        self.normal_init(out_dim, in_dim, mean, var) # Weights
        self.normal_init(out_dim, 1, mean, var, True) # Bias
      else:
        self.random_init(out_dim, in_dim, min, max) # Weights
        self.random_init(out_dim, 1, min, max, bias=True) # Bias

  def normal_init(self, out_dim, in_dim, mean=0.0, var=0.05, bias=False):
    '''
    Initialize parameters by sampling from a normal distribution.

    Args:


    '''
    # If variance not set, use Kaiming's initializaion with fan-in

    # Initialize weights or bias
    if bias:
      self.b = np.random.normal(mean, var**0.5, (out_dim, in_dim))
    else:
      self.w = np.random.normal(mean, var**0.5, (out_dim, in_dim))

  def random_init(self, out_dim, in_dim, min=-0.1, max=0.1, bias=False):
    '''
    Initialize parameters by sampling from a uniform distribution.

    Args:

    '''
    if bias:
      self.b = np.random.uniform(min, max, size=(out_dim, in_dim))
    else:
      self.w = np.random.uniform(min, max, size=(out_dim, in_dim))

  def zero_bias(self):
    '''
    Set layer biases to zero.

    '''
    self.b = np.zeros(self.b.shape)
  
  def forward(self, input, batch_size):
    '''
    Pass inputs forward through the layer.
    
    Args:
      input (ndarray): Feature values to pass through the model
      batch_size (int): Batch size of the input

    '''
    self.z = self.w @ input + self.b # Aggregate
    self.a = (self.activ_fun(self.z)).reshape(-1, batch_size) # Activate
  
  def adjust_weight(self, prev_a, grad_chain, lr, batch_size):
    '''
    Update weight values using the backpropagated gradient.
    
    Args:
      prev_a (ndarray): Activation from the previous layer
      grad_chain (ndarray): Backpropagated chain of derivatives so far
      lr: Learning rate

    '''
    # Normalize magnitude of adjustment with batch size
    self.w -= lr/batch_size * (grad_chain @ prev_a.T)
  
  def adjust_bias(self, grad_chain, lr, batch_size):
    '''
    Update bias values using the backpropagated gradient.

    Args:
      grad_chain (ndarray): Backpropagated chain of derivatives so far
      lr (float): Learning rate
      batch_size (int): Batch size

    '''
    # Derivative of an aggregate wrt bias (dz/db) = 1, so don't need 
    # activation values of the previous layer
    self.b -= lr/batch_size * np.sum(grad_chain, axis=1).reshape(-1, 1)
