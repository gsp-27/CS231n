import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_train = X.shape[1]
  num_classes = W.shape[0]

  for i in range(num_train):
    weighted_sum = W.dot(X[:, i])
    weighted_sum -= np.max(weighted_sum)

    true = weighted_sum[y[i]]
    norma = np.exp(weighted_sum)
    norma_sum = np.sum(norma)
    log_loss = -np.log(np.exp(true) / norma_sum)
    loss += log_loss

    # Code for gradients
    for j in range(num_classes):
      to_add = (norma[j] * X[:, i]) / norma_sum
      
      if j == y[i]:
        # true value gradient is different than the others
        dW[y[i]] += (-1 * X[:, i]) + to_add

      else:
        dW[j] += to_add

  loss = loss / num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW = dW / num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[1]
  num_classes = W.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  # Compute the weighted sum
  weighted_sum = W.dot(X) # dimension is 10 * 49000

  # subtract max of each column from the column
  weighted_sum = weighted_sum - weighted_sum.max(axis=0)

  # extract out true value from each column
  idx = range(num_train)
  true_value = weighted_sum[y, idx]

  # exponentiate each column
  exp_col = np.exp(weighted_sum)
  # sum each column
  sum_exp_col = np.sum(exp_col, axis=0)

  # add true value and sum of each columns and sum them both
  loss = np.sum(-1 * true_value + np.log(sum_exp_col))

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  # Calculation for the gradient
  tiled_sum_exp_col = np.tile(sum_exp_col, (num_classes, 1))
  normalised_expo_weight = exp_col / tiled_sum_exp_col
  
  # Add -1 to the true set of exponentiated weights
  normalised_expo_weight[y, range(num_train)] -= 1
  dW = normalised_expo_weight.dot(X.T)
  
  dW = dW / num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
