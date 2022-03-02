import numpy as np
import sys
import gzip
sys.path.insert(0, r"C:\Users\RrbDellDesktop3\Documents\school\CS_637\hw1\mlp_api")
from mlp_api import *

def read_gz(file, img_size=28, labels=False):
    f = gzip.open(file,'r')
    if labels:
        f.read(8)
        buffer = f.read()
        return np.frombuffer(buffer, dtype=np.uint8)
    else:
        f.read(16)
        buffer = f.read()
        return np.frombuffer(buffer, dtype=np.uint8).reshape(-1, img_size, img_size)

def make_dataset(width=28):
      train_X = read_gz('mnist/train-images-idx3-ubyte.gz')
      train_y = read_gz('mnist/train-labels-idx1-ubyte.gz', labels=True)
      test_X = read_gz('mnist/t10k-images-idx3-ubyte.gz')
      test_y = read_gz('mnist/t10k-labels-idx1-ubyte.gz', labels=True)
      mnist_all_X = np.concatenate((train_X, test_X), axis=0)
      mnist_all_y = np.concatenate((train_y, test_y), axis=0)
      mnist_set = Dataset(mnist_all_X, mnist_all_y)
      mnist_set.shape(width, 10)
      return mnist_set

if __name__ == '__main__':
      width = 28
      mnist_set = make_dataset()
      mnist_set.shuffle()
      mnist_set.divide()
      print('Splits on training, validation, and testing:', 
            mnist_set.X_train.shape, mnist_set.X_valid.shape, mnist_set.X_test.shape)   

      # Test making a model
      batch = 100
      epochs = 10
      dims = (width*width, 128, 64, 10)
      activ_fns = (Activation.relu, Activation.relu, Activation.softmax)

      # Make a generic neural network with two hidden layers
      p_128_64 = Perceptron(dims, activ_fns, Loss.cross_entropy)

      # Different learning rates to test
      lr_results = {}
      lrs = (5e-4, 0.002, 0.01)
      for lr in lrs:
            p_128_64.reset() # Reinitialize parameters
            lr_results[lr] = Utility.train_epochs(p_128_64, mnist_set, lr, epochs, batch)
            break # TEMP
