import numpy as np

class Activation(object):
  '''
  Common activation functions used in a neural network.

  '''

  @staticmethod
  def relu(z, derive=False):
    '''
    Applies ReLU or its derivative on an input.
    
    Args:
      x (ndarray): Input data to apply ReLU on
      derive (bool): True to take derivative of ReLU wrt to X

    Returns:
      Activated or gradient value from applying ReLU.

    '''
    if not derive:
      # Third argument modifies in place
      return np.maximum(z, 0., z) 
    else:
      return (z > 0).astype(int)
  
  @staticmethod
  def sigmoid(z, derive=False, threshold=-100.0):
    '''
    Applies sigmoid or its deriative on an input.

    Args:
      x (ndarray): Input data to apply sigmoid on
      derive (bool): True to take derivative of sigmoid wrt to X
      threshold (float): Minimum threshold to use the first formulation of
        sigmoid for. If any of the inputs are less than this, use second
        formulation to help prevent overflow.

    Returns:
      Activated or gradient value from applying sigmoid

    '''
    # Condition to help prevent numerical overflow or divide by 0 with exponent
    if not derive:
      if (z < threshold).any():
        return np.exp(z) / (np.exp(z) + 1)
      else:
        return 1 / (1 + np.exp(-z))
    else:
      return Activation.sigmoid(z) * (1-Activation.sigmoid(z))

  @staticmethod
  def identity(x, derive=False):
    '''
    Applies the identity function or its derivative on a set inputs.

    Args:
      x (ndarray): Input data to apply sigmoid on
      derive (bool): True to take derivative of sigmoid wrt to X
    
    Returns:
      Same input ("activated") or gradient value from applying the identity function    

    '''
    if not derive:
      return x
    else:
      return 1

  @staticmethod
  def softmax(z, y=None, derive=False):
    '''
    Applies softmax or its derivative on a set of inputs, typically 
    the aggregates from the last layer.

    Args:
      z (ndarray): Input array to apply softmax on
      derive (bool): True to take derivative of sigmoid wrt to z

    Returns: 
      Probabilities for a k-class classification so the sum for a given example
      over all classes = 1.0, or the gradient of softmax
      
    '''
    # Update from using a naive implementation - prevent numerical overflows
    # by using 'shifting' trick as opposed to np.exp(x)/np.sum(np.exp(x), axis=0)
    shift = np.max(z)
    exp_z = np.exp(z - shift)
    yhat = exp_z/np.sum(exp_z, axis=0)

    if not derive:
      return yhat
    else:
      assert y is not None

      # Need to know what class is the correct class
      yhat_target = np.sum(np.where(y == 1, yhat, 0), axis=0)

      # Derivative of softmax when i != j (where j is the target class) is
      # -yhat_i*yhat_j. Derivative when i == j is yhat_j*(1 - yhat_j)
      return np.where(y != 1, -yhat_target.T*yhat, yhat_target.T*(1 - yhat_target))