import numpy as np
import sys
import time
sys.path.insert(0, r"C:\Users\RrbDellDesktop3\Documents\school\CS_637\hw1\mlp_api")
from mlp_api import *

# Test with dummy data
width = 28
categories = 10
mnist_all_X = np.random.normal(size=(10, width, width))
mnist_all_y = np.arange(10)

mnist_set = Dataset(mnist_all_X, mnist_all_y)
mnist_set.shape(width, 10)
#Utility.plot_images(mnist_set, 5, show=True)

print(mnist_set.X.shape[-1], mnist_set.y.shape[-1])
mnist_set.shuffle()
mnist_set.divide()
print('Splits on training, validation, & testing:', 
      mnist_set.X_train.shape, mnist_set.X_valid.shape, mnist_set.X_test.shape)


# Test making a model
# Hyperparameters to use throughout all models
batch = 1
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
  lr_results[lr] = Utility.train_epochs(p_128_64, mnist_set, 
                                              lr, epochs, batch)
