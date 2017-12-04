from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # pass
        C, H, W = input_dim
        F = num_filters
        HH = WW = filter_size
        self.params['W1'] = np.random.normal(scale=weight_scale, size=[F, C, HH, WW])
        self.params['b1'] = np.zeros([F])
        self.params['W2'] = np.random.normal(scale=weight_scale, size=[F*H//2*W//2, hidden_dim])
        self.params['b2'] = np.zeros([hidden_dim])
        self.params['W3'] = np.random.normal(scale=weight_scale, size=[hidden_dim, num_classes])
        self.params['b3'] = np.zeros([num_classes])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # pass
        
        N, C, H, W = X.shape
        F = W1.shape[0]
        
        
        hidden_dim = W2.shape[1]
        # conv + relu + maxpool
        conv_out, conv_cache = conv_forward_naive(X, W1, b1, conv_param)
        conv_relu_out, conv_relu_cache = relu_forward(conv_out)
        conv_pool_out, conv_pool_out_cache = max_pool_forward_naive(conv_relu_out, pool_param)
        
        # reshape
        conv_pool_out_reshape = conv_pool_out.reshape([-1, F*H//2*W//2])
        # affine + relu
        affine_out_1, affine_out_1_cache = affine_forward(conv_pool_out_reshape, W2, b2)
        affine_relu_out_1, affine_relu_out_1_cache = relu_forward(affine_out_1)
        # affine
        affine_out_2, affine_out_2_cache = affine_forward(affine_relu_out_1, W3, b3)
        
        scores = affine_out_2
        
        #print(conv_out.shape)
        #print(conv_relu_out.shape)
        #print(conv_pool_out.shape)
        #print(conv_pool_out_reshape.shape)
        #print("  ")
        #print(affine_out_1.shape)
        #print(affine_relu_out_1.shape)
        #print("  ")
        #print(affine_out_2.shape)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        # pass
        
        
        loss, dscores = softmax_loss(scores, y)
        l2_loss = 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
        loss += l2_loss
        #print("  ")
        #print(dscores.shape)
        daffine_out_2, dW3, db3 = affine_backward(dscores, affine_out_2_cache)
        #print(daffine_out_2.shape)
        
        daffine_relu_out_1 = relu_backward(daffine_out_2, affine_relu_out_1_cache)
        daffine_out_1, dW2, db2 = affine_backward(daffine_relu_out_1, affine_out_1_cache)
        
        #print("  ")
        #print(daffine_relu_out_1.shape)
        #print(daffine_out_1.shape)
        
        daffine_out_1_reshape = daffine_out_1.reshape([N, F, H//2, W//2])
        #print("  ")
        #print(daffine_out_1_reshape.shape)
        dconv_pool_out = max_pool_backward_naive(daffine_out_1_reshape, conv_pool_out_cache)
        dconv_relu_out = relu_backward(dconv_pool_out, conv_relu_cache)
        dconv_out, dW1, db1 = conv_backward_naive(dconv_relu_out, conv_cache)
        
        grads['W1'] = dW1 + self.reg * W1
        grads['W2'] = dW2 + self.reg * W2
        grads['W3'] = dW3 + self.reg * W3
        grads['b1'] = db1
        grads['b2'] = db2
        grads['b3'] = db3
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
