import numpy as np
import matplotlib.pyplot as plt
import time
import os, sys
sys.path.insert(0, os.path.join(os.getcwd() + r'/mlp_api/mlp_api'))
import loss

class Utility(object):
  '''
  Miscellaneous methods for training and plotting.

  '''

  @staticmethod
  def train_epochs(mlp, dataset, lr, epochs, batch_size,
                   shuffle_every_epoch=True, verbose=True, hinge_and_logits=False):
    '''
    Train a model over several epochs.

    Args:
      mlp: Perceptron object to train
      dataset: Dataset object to pull features and labels from
      lr (float): Learning rate
      epochs (int): Number of epochs to train in
      batch_size (int): Batch size
      shuffle_every_epoch (bool): True to reshuffle training and validation
        data within their divisions when regenerating the batches
      verbose (bool): True to print loss and accuracy values after each epoch

    Returns:
      Array of losses at each epoch, array of accuracy at each epoch
    
    '''
    assert (epochs > 0)

    train_loss = np.empty(epochs)
    train_acc = np.empty(epochs)
    start_time = time.time()
    if verbose:
      print('Epoch \tTrain_loss \tTrain_acc')

    for epoch in range(epochs):
      train_iter = dataset.make_batches(batch_size, group='train',
        shuffle_again=shuffle_every_epoch)
      train_loss[epoch], train_acc[epoch] = mlp.pass_data(train_iter, lr,
        batch_size, train_mode=True, hinge_and_logits=hinge_and_logits)
      
      # If set to print out info as executing
      if verbose:
        print('{} \t{:9.3f} \t{:9.3f}'.format(
            epoch, train_loss[epoch], train_acc[epoch]))
      
      # Check for explosion or vanishing
      if np.isnan(train_loss[epoch]) or np.isinf(train_loss[epoch]):
        raise Exception('Loss has become NaN or infinity! Stop training.')

    # Print amount of time taken
    print('Time elapsed (s): {:2.1f}\n'.format(time.time() - start_time))


    return train_loss, train_acc
  
  @staticmethod
  def make_one_hot(y):
    '''
    Make a traditional container of labels into a one-hot representation.
    
    Args:
      y (list or tuple): Container of labels to convert

    Returns:
      Transformed array in one-hot representation
    
    '''
    transformed = np.zeros((y.size, y.max() + 1))
    transformed[np.arange(y.size), y] = 1
    return transformed.T

  @staticmethod
  def reverse_one_hot(y):
    '''
    Convert a one-hot array back to a simple list of indices.

    Args:
      y (ndarray): Array to convert
    
    Returns:
      Transformed array as a sequence of indexes

    '''
    return np.argmax(y, axis=0)

  @staticmethod
  def plot_results(results, labels=None, fmts=None, xlabel='Epoch', ylabel='',
                   ymax=None, title=''):
    
    '''
    Plot one or more results

    Args:
      results (list or tuple): Container of ndarray results to plot
      labels (list or tuple): Container of strings, each a label corresponding
        to each result to be displayed in the legend
      fmts (list or tuple): Container of strings matching the matplotlib's
        line, color, and marker styles. None to use default
      xlabel (str): Label on x-axis
      ylabel (str): Label on y-axis
      ymax (float): Maximum bound to set y-axis to. None to use automatic
        settings
      title (str): Title of plot

    '''
    # Account for arguments not given
    if labels is None: 
      labels = (str(_) for _ in range(len(results)))
    else:
      assert len(results) == len(labels)

    if fmts is None:
      fmts = ('' for _ in range(len(results)))
    else:
      assert len(results) == len(fmts)

    # Plot each set of results with its corresponding label and format
    for result, label, fmt in zip(results, labels, fmts):
      plt.plot(np.arange(result.shape[0]), result, fmt, label=label)

    # Parameters affecting entire plot
    plt.legend(framealpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)  
    if ymax is not None:
      plt.ylim(0, ymax)

  @staticmethod
  def plot_images(data, images=3, show_random=False, mlp=None, cols=5,
                  show=False):
    '''
    Plot using the flattened representation of a square image in input X.

    Args:
      data: Dataset object to select a random sample from
      images (int): Number of images to sample and show
      mlp: Perceptron object to pass images through. None to not make any
        predictions and just show the true labels instead.
      cols (int): Number of columns to arrange images in
      show (bool): True to explicitly call plt.show() - do not need to set
        if running in a Jupyter notebook

    '''
    if show_random: # Get random sample from dataset
      indexes = np.random.randint(0, data.X.shape[-1], images)
    else: # Get first images in range from dataset
      indexes = tuple(range(images))

    sample_X, sample_y = data.X[:, indexes], data.y[:, indexes]
    sample_y = Utility.reverse_one_hot(sample_y)

    # Pass selected images through MLP to get predictions, if requested
    if mlp is not None:
      mlp.forward(sample_X, sample_X.shape[-1])
      sample_yhat = loss.Loss.accuracy(mlp[-1].a, sample_y, False, True)

    # Display data as images
    width = int(data.X.shape[0]**0.5)
    for i in range(images):  
      plt.subplot(np.ceil(images/cols).astype(int), cols, i + 1)
      plt.axis('off')
      plt.imshow(sample_X[:, i].reshape(width, width), 
                 cmap=plt.get_cmap('gray'))
      
      # Title each image with its true label      
      if mlp is not None:
          plt.title('true: {}\npredict: {}'.format(sample_y[i], sample_yhat[i]))
      else:
          plt.title('true: {}'.format(sample_y[i]))

    plt.subplots_adjust(hspace=1.0)
    
    if show:
      plt.show()
