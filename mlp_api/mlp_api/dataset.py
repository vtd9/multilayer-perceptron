import numpy as np
import os, sys

from pyrsistent import v
sys.path.insert(0, os.path.join(os.getcwd() + r'/mlp_api/mlp_api'))
import utility

class Dataset(object):
  '''
  Represents a set of data with methods to reshape, split, and train.

  '''
  def __init__(self, X, y, make_y_one_hot=True, scale_X=True, 
               max_intensity=255.):
    '''
    Constructs a Dataset object.
    
    Args:
      X (ndarray): Input data
      y (ndarray): Labels corresponding to input data
      make_y_one_hot (bool): True to convert the labels into a one-hot 
        representation, if not done already; false to skip this step
      scale_X (bool): True to convert image inputs to be between 0 and 1
      max_intensity (float): Maximum intensity of input. Default = 255.
    '''
    self.X = X
    self.y = y.astype('int32')
    self.X_train = None
    self.y_train = None
    self.X_valid = None
    self.y_valid = None
    self.X_test = None
    self.y_test = None

    if make_y_one_hot:
      self.y = utility.Utility.make_one_hot(self.y)
    if scale_X:
      self.X = self.X.astype('float64') / max_intensity

  def shape(self, features, categories, flatten_X=True):
    '''
    Reshapes the input and output data.

    Args:
      features: Number of features in the input data. Set to width of a 2D image
        if it will be flattened.
      categories: Number of categories in the classification problem
      flatten_X: True to flatten a 2D image into a 1D array

    '''
    if flatten_X: 
      self.X = self.X.reshape(-1, features*features).T
    else:
      self.X = self.X.reshape(features, -1)
    self.y = self.y.reshape(categories, -1)

  def shuffle(self):
    '''
    Shuffles both X and y together in the dataset.

    '''
    assert self.X.shape[-1] == self.y.shape[-1]
    shuffled_indexes = np.random.permutation(self.X.shape[-1])
    self.X = self.X[:, shuffled_indexes]
    self.y = self.y[:, shuffled_indexes]

  def divide(self, p_train=70, p_valid=15, p_test=15):
    '''
    Divide the loaded data into sets for training, validation, and testing.

    Args:
      p_train (int): Percentage of data to allot for training
      p_valid (int): Percentage of data to allot for validation
      p_test (int): Percentage of data to allot for testing

    '''
    if (p_train + p_valid + p_test != 100):
      raise ValueError('Error: percentages don''t sum up to 100!')
    n_train = int(p_train * 0.01 * self.X.shape[-1])
    n_valid = int(p_valid * 0.01 * self.X.shape[-1])
    n_test = int(p_test * 0.01 * self.X.shape[-1])

    # If missing a few examples from cutoffs, add to training set
    test_diff = self.X.shape[-1] - (n_train + n_valid + n_test)
    if test_diff > 0:
      n_train += test_diff

    # Define groups
    self.X_train = self.X[:, 0:n_train]
    self.y_train = self.y[:, 0:n_train]
    self.X_valid = self.X[:, n_train:(n_train + n_valid)]
    self.y_valid = self.y[:, n_train:(n_train + n_valid)]
    self.X_test = self.X[:, (n_train + n_valid): ]
    self.y_test = self.y[:, (n_train + n_valid): ]
    
  def make_batches(self, batch_size, group='train', shuffle_again=True):
    '''
    Make batches given a specified size and a selected group.

    Args: 
      batch_size (int): Batch size to divide data into
      group (str): Either train, valid, or test to select group
      shuffle_again (bool): True to shuffle data within its group when 
        a new set of batches is made

    Returns:
      Generator of batches in the specified size from the specified group.

    '''
    # Get the training, testing, or validation group:
    if group == 'train':
      X_select, y_select = self.X_train, self.y_train
    elif group == 'valid':
      X_select, y_select = self.X_valid, self.y_valid
    elif group == 'test':
      X_select, y_select = self.X_test, self.y_test
    else:
      raise ValueError('Incorrect argument for group! Choose between train, '
                       'validate, or test.')

    # Check group divisible by batch size requested
    if X_select.shape[-1] % batch_size != 0:
      raise Exception('Group not divisible by batch size! Choose another batch size.')

    # Shuffle within the group, if specified
    if shuffle_again:
      shuffled_indexes = np.random.permutation(X_select.shape[-1])
      X_select = X_select[:, shuffled_indexes]
      y_select = y_select[:, shuffled_indexes]

    # Make generators for the features and labels of the selected set
    for i in range(0, X_select.shape[-1], batch_size):
      yield X_select[:, i:i+batch_size], y_select[:, i:i+batch_size]
