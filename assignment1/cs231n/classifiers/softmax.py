import numpy as np
from random import shuffle
from past.builtins import xrange

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
  # pass
  num_train = X.shape[0]
  num_class = W.shape[1]
  dim_input = X.shape[1]
  
  for i in range(num_train):
    f_xi = X[i].dot(W)
    stablized = f_xi - np.max(f_xi)
    exp = np.exp(stablized)
    sum = np.sum(exp)
    out = exp / sum
    loss += -np.log(out[y[i]])
    
    grouth_truth = np.zeros(num_class)
    grouth_truth[y[i]] = 1
    grad_out = out - grouth_truth
    grad_out = grad_out.reshape([1, num_class])
    dW += X[i].T.reshape([dim_input, 1]).dot(grad_out)
    
    
    
  loss = loss / num_train + 0.5 * reg * np.sum(W * W)
  dW = dW / num_train + reg * W
    
  
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
  # pass
  num_train = X.shape[0]
  num_class = W.shape[1]
  dim_input = X.shape[1]

  scores = X.dot(W)
  stablized = scores - np.max(scores, axis=1).reshape([num_train, 1])
  exp = np.exp(stablized)
  sum = np.sum(exp, axis=1).reshape([num_train, 1])
  prob = exp / sum
  loss = np.sum(-np.log(prob)[np.arange(num_train), y])
  
  ground_truth = np.zeros([num_train, num_class])
  ground_truth[np.arange(num_train), y] = 1
  
  # dscores is the gradient that \delta{Loss} / \delta{scores}
  dscores = prob - ground_truth
  dW = X.T.dot(dscores)

  loss = loss / num_train + float(0.5 * reg * np.sum(W ** 2))
  dW = dW / num_train + reg * W
   

  
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

