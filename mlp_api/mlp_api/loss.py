import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.getcwd() + r'/mlp_api/mlp_api'))
import utility

class Loss(object):
  '''
  Common loss functions used in a neural network.

  '''

  @staticmethod
  def cross_entropy(yhat, y, epsilon=1e-5, derive=False):
    '''
    Computes the cross-entropy loss or its derivative with respect to
    the model outputs.

    Args:
      yhat (ndarray): Outputs (after activation) from a MLP
      y (ndarray): Actual labels to compare output with
      epsilon (float): Small positive value to prevent taking log of 0
      derive (bool): True to return derivative wrt outputs

    Returns:
      Cross-entropy loss value or its derivative wrt outputs

    '''
    if not derive:
      # Squash for each example by summing along axis 0 before averaging - 
      # erroneously many zeros otherwise
      return np.average(np.sum(-y * np.log(yhat + epsilon), axis=0))
    else:
      # Squash active yhat values down by summing down the columns
      return np.sum(-y / yhat, axis=0).reshape(-1, 1).T

  @staticmethod
  def hinge_loss(z, y, margin=1.0, derive=False):
    '''
    Computes the hinge loss or its derivative for a batch of predictions.

    Args:
      z (ndarray): Logits (outputs before softmax) from a MLP, or otherwise
        any output to measure the performance of
      y (ndarray): Actual labels to compare output with
      margin (float): Margin for target class value to overtake
      derive (bool): True to return derivative

    Returns:
      Hinge loss value or its derivative

    '''
    # Output of the target class for each example
    # Repeat so same value going down row to get same shape as yhat
    y_target = np.sum(np.where(y == 1, z, 0), axis=0)

    # Get distances from outputs at indexes != target class
    # Add margin to elements at indexes != target class
    dist = np.where(y == 1, 0, z - y_target + margin)

    if not derive:
      # Get the positive differences over each example (zero-threshold)
      pos_per_example = np.where(dist > 0, dist, 0)

      # Sum over each example, average over all data points
      return np.average(np.sum(pos_per_example, axis=0))
    else:
      return np.where(y == 1, -np.sum(dist > 0, axis=0), dist > 0)
  
  @staticmethod
  def accuracy(yhat, y, y_one_hot=True, return_predict=False):
    '''
    Computes the accuracy for a batch of outputs.

    Args:
      yhat (ndarray): Outputs (after activation) from a MLP
      y (ndarray): Actual labels to compare output with
      y_one_hot: True if y given in one-hot form, false if given as a list 
        of correct indexes

    Returns:
      Accuracy of a given output against its true values

    '''
    # Prediction class is the one with max predicted probability in each example
    yhat_choices = np.argmax(yhat, axis=0)

    # Reverse the actual labels' representation as one-hot for easier comparison
    if y_one_hot:
      y = utility.Utility.reverse_one_hot(y)

    # If want to return the actual predictions
    if return_predict:
      return yhat_choices

    # Else, return the average accuracy
    return np.average(yhat_choices == y)
