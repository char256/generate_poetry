import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
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
  y_pred = X.dot(W)
  y_pred_exp = np.exp(y_pred)
  y_pred_sum = np.sum(y_pred_exp,axis = 1)
  loss = np.sum(-np.log(y_pred_exp[range(X.shape[0]),y]/y_pred_sum))/X.shape[0] \
          + 0.5 * reg * np.sum(W*W)
  for i in xrange(X.shape[0]):
    scores = X[i].dot(W)
    scores_exp = np.exp(scores)
    ds = scores_exp / np.sum(scores_exp)
    ds[y[i]] -= 1
    for j in xrange(W.shape[1]):
      dW[:,j] += ds[j] * X[i]
  dW /= X.shape[0]
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  y_pred = X.dot(W)
  y_pred_exp = np.exp(y_pred)
  y_pred_sum = np.sum(y_pred_exp,axis = 1)
  loss = np.sum(-np.log(y_pred_exp[range(X.shape[0]),y]/y_pred_sum))/X.shape[0] \
          + 0.5 * reg * np.sum(W*W)
  scores =X.dot(W)
  scores_exp = np.exp(scores)
  ds = scores_exp.T / np.sum(scores_exp,axis = 1)
  #print "ds.shape=",ds.shape
  ds = ds.T
  ds[range(y.shape[0]),y] -= 1
  #print "ds.shape=",ds.shape
  dW = X.T.dot(ds)
  dW /= X.shape[0]
  #print dW.shape,W.shape,ds.shape
  #print scores.shape,scores_exp.shape
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

