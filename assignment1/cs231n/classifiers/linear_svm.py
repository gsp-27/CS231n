import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[j] += X[:, i]
        dW[y[i]] -= X[:, i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, (dW / num_train)


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  delta = 1.0
  num_train = X.shape[1]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = W.dot(X)
  idx = range(X.shape[1])
  correct_score = scores[y, idx]

  # print scores[y[0], 0], correct_score[0]
  
  correct_score = np.tile(correct_score, (10,1))
  loss = np.sum(np.maximum(np.zeros((W.shape[0], X.shape[1])), scores - correct_score + delta))
  loss -= X.shape[1] * delta
  loss /= X.shape[1]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # Calculate 0, loss maximum
  # take out all the non-zero entries
  # multiply with the training examples matrix transpose.
  # Add the resulting ans to dW
  maximum_mask = np.maximum(np.zeros((W.shape[0], X.shape[1])), scores - correct_score + delta)
  maximum_mask[y, idx] = 0

  maximum_mask[maximum_mask != 0] = 1
  
  sum_columnwise = np.sum(maximum_mask, axis=0)
  # replace correct entry with sum of columns
  maximum_mask[y, idx] = -sum_columnwise[range(num_train)]

  # Here we are doing two things at once, first we are calculating sum of all 1 entries in row
  # and then subtract that many number of times as sum of ones across column.
  dW = maximum_mask.dot(X.T)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW