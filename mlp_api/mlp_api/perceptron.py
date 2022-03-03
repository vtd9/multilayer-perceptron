import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.getcwd() + r'/mlp_api/mlp_api'))
import layer, loss

class Perceptron(object):
  '''
  Represents a multi-layer perceptron (MLP) for processing input data. 
  
  Made of an aggregate of Layer objects that can be accessed from indexes,
  where 0 is the input layer, 1 is the first hidden layer, and so forth.

  '''

  def __init__(self, dims, activ_fns, loss_fn,
               init_with_normal=True, mean=0.0, var=0.05, min=-0.1, max=0.1):
    '''
    Constructs a new Perceptron object.

    Args:
      dims (list or tuple): Container of ints, each representing the 
        dimensionality of a layer at the same index. First integer should be
        the number of input features; last int is the number of output neurons.
      activ_fns (list or tuple): Container of methods from the Activation class 
      to apply on the hidden and output aggregations
      loss_fn: loss function to measure model performance
      init_with_normal (bool): True to initialize parameters by sampling from a 
        normal distribution, false to sample from a uniform one
      mean (float): Mean to use if initializing with a normal dist
      var (float): Variance to use if initializing with a normal dist
      min (float): Minimum bound to use if initializing with a uniform dist
      max (float): Minimum bound to use if initializing with a uniform dist

    '''
    # Ensure Perceptron has at least input and output layers
    assert len(dims) > 1

    # Ensure number of activation functions matches number of layers - 1
    assert len(dims) == len(activ_fns) + 1
    
    self.dims = dims
    self.n_layers = len(dims)
    self.activ_fns = activ_fns
    self.loss_fn = loss_fn
    self.reset(init_with_normal, mean, var, min, max) # Initialize parameters

  def reset(self, init_with_normal=True, mean=0.0, var=0.05, min=-0.1, max=0.1):  
    '''
    Initialize parameters in the MLP.

    Args:


    '''
    self.layers = [None] * len(self.dims)

    # For each layer besides the input layer, attach the appropriate dimensions, 
    # activation function, and initialization parameters
    for i in range(self.n_layers - 1):
      self.layers[i + 1] = layer.Layer(
          in_dim=self.dims[i], out_dim=self.dims[i+1],
          activ_fun=self.activ_fns[i], 
          init_with_normal=init_with_normal,
           mean=mean, var=var, min=min, max=max)
  
  def zero_biases(self):
    '''
    Set all biases in the MLP to zero.

    '''
    for i in range(1, self.n_layers):
      self.layers[i].zero_bias()

  def __getitem__(self, layer_index):
    '''
    Retrieves reference to a layer in the MLP.

    Args:
      layer_index (int): Index of layer to access.

    Returns: 
      Layer object at layer_index

    '''
    return self.layers[layer_index]
  
  def forward(self, X, batch_size):
    '''
    Step forward through the MLP, aggregating the previous layer's values and
    then applying the chosen activation function.

    Args:
      X (ndarray): Input data to feed through the network
      batch_size (int): Batch size

    '''
    # Set the first layer (the input layer)'s "activation" to the input data
    self.layers[0] = layer.Layer(activated=X)

    # Pass through each layer using inputs from previous layer
    for i, layer_ in enumerate(self.layers[1:], start=1):
      layer_.forward(self.layers[i-1].a, batch_size)
  
  def backward(self, labels, lr, batch_size, hinge_and_logits=False):
    '''
    Backpropagates from loss and adjust parameters after a forward pass.
    
    Args:
      labels (ndarray): True labels in a one-hot representation
      lr (float): Learning rate
      batch_size (int): Batch size
    
    '''
    # Calculate derivative of the chosen loss function wrt outputs
    # If hinge loss applied on logits (before output activation), use appropriate z
    if (hinge_and_logits):
      assert 'hinge_loss' in str(self.loss_fn)
      grad_chain = self.loss_fn(self.layers[-1].z, labels, derive=True)
    else:
      grad_chain = self.loss_fn(self.layers[-1].a, labels, derive=True)

      # If activation was softmax, need to provide labels, too
      if 'softmax' in str(self.activ_fns[-1]):
        grad_chain = grad_chain * self.activ_fns[-1](self.layers[-1].z, labels, derive=True)
      else:
        grad_chain = grad_chain * self.activ_fns[-1](self.layers[-1].z, derive=True)
    
    for i in range(self.n_layers - 1, 0, -1):
      # Save current weight for dloss/da of the previous layer later
      prev_a = self.layers[i-1].a
      old_weight = self.layers[i].w

      # Adjust parameters
      self.layers[i].adjust_weight(prev_a, grad_chain, lr, batch_size)
      self.layers[i].adjust_bias(grad_chain, lr, batch_size)

      # Compute dloss/da of previous layer, multiply to current product
      grad_chain = (old_weight.T @ grad_chain) 

      # Compute dloss/dz of previous layer using elementwise multiplication
      if self.layers[i-1].activ_fun is not None:
        da_dz = self.layers[i-1].activ_fun(prev_a, derive=True)
      else: # Don't need to derive the input layer
        break
      grad_chain *= da_dz
  
  def pass_data(self, generator, lr=0.001, batch_size=100, train_mode=True,
    hinge_and_logits=False):
    '''
    Pass data through the model in a single epoch.

    Args:
      generator (generator): Generator to loop through one batch at a time
      lr (float): Learning rate
      train_mode (bool): True to backpropagate and update parameters in training
    
    Returns:
      Average loss (float), average accuracy (float)

    '''
    loss_epoch = 0.0
    acc_epoch = 0.0
    for i, (X, y) in enumerate(generator):
      self.forward(X, batch_size)

      if train_mode: # Adjust parameters only during training
        self.backward(y, lr, batch_size, hinge_and_logits)

      # Update accumulated measures
      loss_epoch += self.loss_fn(self[-1].a, y)
      acc_epoch += loss.Loss.accuracy(self[-1].a, y)

    # Return average loss and accuracy over the number of batches done
    return loss_epoch/(i+1), acc_epoch/(i+1)