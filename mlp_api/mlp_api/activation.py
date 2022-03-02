import numpy as np

class Activation(object):
  '''
  Common activation functions used in a neural network.

  '''

  @staticmethod
  def relu(x, derive=False):
    '''
    Applies ReLU or its derivative on an input.
    
    Args:
      x (ndarray): Input data to apply ReLU on
      derive (bool): True to take derivative of ReLU wrt to X

    Returns:
      Activated or gradient value from applying ReLU.

    '''
    if not derive:
      return np.maximum(x, 0., x)
    else:
      return (x > 0).astype(int)
  
  @staticmethod
  def sigmoid(x, derive=False, threshold=-100.0):
    '''
    Applies sigmoid or its deriative on an input.

    Args:
      x (ndarray): Input data to apply sigmoid on
      derive (bool): True to take derivative of sigmoid wrt to X
      threshold (float): Minimum threshold to use the first formulation of
        sigmoid for. If any of the inputs are less than this, use second
        formulation to help prevent overflow.

    Returns:
      Activated or gradient value from applying sigmoid.

    '''
    # Condition to help prevent numerical overflow or divide by 0 with exponent
    if not derive:
      if (x < threshold).any():
        return np.exp(x) / (np.exp(x) + 1)
      else:
        return 1 / (1 + np.exp(-x))
    else:
      return Activation.sigmoid(x) * (1-Activation.sigmoid(x))


  @staticmethod
  def softmax(z, y=None, derive=False):
    '''
    Args:
      x (ndarray): Input array to apply softmax on

    Returns: 
      Probabilities for a k-class classification so the sum for a given example
      over all classes = 1.0
      
    '''
    # Update from using a naive implementation - prevent numerical overflows
    # by using shifting trick as opposed to np.exp(x)/np.sum(np.exp(x), axis=0)
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