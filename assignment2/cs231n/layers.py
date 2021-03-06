from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # pass
    N = x.shape[0]
    _x = x.reshape([N, -1])
    out = np.dot(_x, w) + b
    # print(x.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # pass
    # print(x.shape)
    N = x.shape[0]
    x_ = x.reshape([N, -1])
    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(x_.T, dout)
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # pass
    out = x * (x > 0)
    # print(out.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # pass
    dx = dout * 1. * (x > 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        # pass
        sample_mean = np.mean(x, axis=0)
        _minus = x - sample_mean
        _square = _minus ** 2
        sample_var = np.mean(_square, axis=0)
        _sqrt = np.sqrt(sample_var + eps)
        _div = _minus / _sqrt
        _mult = gamma * _div
        out = _mult  + beta
           
        #print("minus ", _minus.shape)
        #print("square ", _square.shape)
        #print("sqrt ", _sqrt.shape)
        #print("div ", _div.shape)
        #print("mult ", _mult.shape)
        #print("sample mean ", sample_mean.shape)
        #print("sample var ", sample_var.shape)
        
        running_mean = momentum * running_mean + (1. - momentum) * sample_mean
        running_var = momentum * running_var + (1. - momentum) * sample_var
        
        cache = (_minus, _square, _sqrt, _div, _mult, sample_mean, sample_var, gamma, beta, eps)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # pass
        out = (x - running_mean) / (np.sqrt(running_var) + eps) * gamma + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    # pass
    _minus, _square, _sqrt, _div, _mult, sample_mean, sample_var, gamma, beta, eps = cache
    
    N, D = dout.shape
    # add ops
    dbeta = np.sum(dout, axis=0)
    _dmult = dout
    
    # mult ops
    dgamma = np.sum(_dmult * _div, axis=0)
    _ddiv = _dmult * gamma
    
    #div ops
    _dminus1 = _ddiv / _sqrt
    _dsqrt = np.sum(_ddiv * (-1 * _minus / (_sqrt ** 2)), axis=0)
    
    #sqrt ops
    dsample_var = _dsqrt * 0.5 / np.sqrt(sample_var + eps)
    
    #mean ops
    _dsquare = dsample_var / N
    
    #sqaure ops
    _dminus2 = _dsquare * 2 * _minus
    
    #minus path split
    _dminus = _dminus1 + _dminus2
    
    #minus ops
    _dsample_mean = -1 * np.sum(_dminus, axis=0)
    _dx1 = 1 * _dminus
    
    #mean ops
    _dx2 = _dsample_mean / N
    
    #x path split
    dx = _dx1 + _dx2
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_forward_alt(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        # pass
        N, D = x.shape
        sample_mean = np.mean(x, axis=0)
        sample_var = np.mean((x - sample_mean) ** 2, axis=0)
        sqrt_sample_var = np.sqrt(sample_var + eps)
        
        x_norm = (x - sample_mean) / sqrt_sample_var
        out = x_norm * gamma + beta
        
        
        
        running_mean = momentum * running_mean + (1. - momentum) * sample_mean
        running_var = momentum * running_var + (1. - momentum) * sample_var
        
        cache = (x_norm, gamma, sqrt_sample_var)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # pass
        out = (x - running_mean) / (np.sqrt(running_var) + eps) * gamma + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache





def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # pass
    N, D = dout.shape
    x_norm, gamma, sqrt_sample_var = cache
    dgamma = np.sum(x_norm * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx = N * dout * gamma - np.sum(dout * gamma, axis=0) - x_norm * (np.sum(dout * gamma * x_norm, axis=0))
    dx /= N * sqrt_sample_var
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # pass
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # pass
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # pass
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # pass
    
    def im2col(x, hh, ww, stride):
        # x shape [C, H, W]
        C, H, W = x.shape
        out_h = (H - hh) // stride + 1
        out_w = (W - ww) // stride + 1
        out = np.zeros([out_h*out_w, hh*ww*C])
        for h_idx in range(out_h):
            for w_idx in range(out_w):
                out[h_idx*out_h+w_idx, :] = x[:,h_idx*stride:(h_idx*stride+hh),w_idx*stride:(w_idx*stride+ww)].ravel()
        return out
            
    def col2im(x, out_h, out_w):
        # x shape [out_h*out_w, F]
        _, F = x.shape
        out = np.zeros([F, out_h, out_w])
        for idx in range(F):
            out[idx, :, :] = x[:, idx].reshape([out_h, out_w])
        return out
    
    x_padded = np.pad(x, 
                      ((0, 0), (0, 0), 
                      (conv_param['pad'], conv_param['pad']), 
                      (conv_param['pad'], conv_param['pad'])), 
                      mode='constant')
    
    N, C, H, W = x_padded.shape
    F, _, HH, WW = w.shape
    
    w = w.reshape([F, C*HH*WW])
    
    out_h = (H - HH) // conv_param['stride'] + 1
    out_w = (W - WW) // conv_param['stride'] + 1
    
    out = np.zeros([N, F, out_h, out_w])
       
    sample_cols = []
    for sample_idx in range(N):
        # sample_col [out_h*out_w, HH*WW*C]
        x_padded_sample = x_padded[sample_idx]
        sample_col = im2col(x_padded_sample, HH, WW, conv_param['stride'])
        # conv_out [out_h*out_w, F]
        sample_conv_out = sample_col.dot(w.T) + b
        sample_image = col2im(sample_conv_out, out_h, out_w)
        out[sample_idx] = sample_image
        sample_cols.append(sample_col)
        
        
    w = w.reshape([F, C, HH, WW])
        
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, sample_cols, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # pass
    
    def col2im_backward(dout):
        # dout shape [F, out_h, out_w]
        F, out_h, out_w = dout.shape
        dx = dout.reshape([F, out_h*out_w])
        return dx
    
    def im2col_backward(dout, C, H, W, HH, WW, stride):
        # dout [out_h*out_w, HH*WW*C]
        out_h = (H - HH) // stride + 1
        out_w = (W - WW) // stride + 1
        
        #out [C, H, W]
        dx = np.zeros([C, H, W])
        for idx in range(out_h*out_w):
            h_idx = (idx // out_h) * stride
            w_idx = (idx % out_w) * stride
            dpatch = dout[idx].reshape([C, HH, WW])
            dx[:, h_idx:(h_idx+HH), w_idx:(w_idx+WW)] += dpatch
        return dx
            
        
    
    x, w, b, sample_cols, conv_param = cache
    
    x_padded = np.pad(x, 
                      ((0, 0), (0, 0), 
                      (conv_param['pad'], conv_param['pad']), 
                      (conv_param['pad'], conv_param['pad'])), 
                      mode='constant')
    
    N, C, H, W = x_padded.shape
    F, _, HH, WW = w.shape
    
    w = w.reshape([F, C*HH*WW])
    
    out_h = (H - HH) // conv_param['stride'] + 1
    out_w = (W - WW) // conv_param['stride'] + 1
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    for sample_idx in range(N):
        sample_col = sample_cols[sample_idx]
        dsample_image = dout[sample_idx]
        dsample_conv_out = col2im_backward(dout[sample_idx])       
        dsample_col = dsample_conv_out.T.dot(w)
        dw_sample = dsample_conv_out.dot(sample_col)
        dx_padded_sample = im2col_backward(dsample_col, C, H, W, HH, WW, conv_param['stride'])
        dx[sample_idx] = dx_padded_sample[:, conv_param['pad']:H-conv_param['pad'], conv_param['pad']:W-conv_param['pad']]
        dw += dw_sample
    dw = dw.reshape([F, C, HH, WW])
    db = np.sum(dout, axis=(0, 2, 3))
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    #pass
    
    def im2col(x, hh, ww, stride):
        # x shape [C, H, W]
        C, H, W = x.shape
        out_h = (H - hh) // stride + 1
        out_w = (W - ww) // stride + 1
        out = np.zeros([out_h*out_w, C*hh*ww])
        for h_idx in range(out_h):
            for w_idx in range(out_w):
                out[h_idx*out_h+w_idx, :] = x[:,h_idx*stride:(h_idx*stride+hh),w_idx*stride:(w_idx*stride+ww)].ravel()
        out = out.reshape([out_h*out_w, C, hh*ww])
        max_value = np.max(out, axis=2)
        max_index = np.argmax(out, axis=2)
        # max_value [C, out_h, out_w]
        max_value = max_value.reshape([out_h, out_w, C]).transpose(2, 0, 1)
        #print(max_index.shape)
        return max_value, max_index
    N, C, H, W = x.shape
    out_h = (H - pool_param['pool_height']) // pool_param['stride'] + 1
    out_w = (W - pool_param['pool_width']) // pool_param['stride'] + 1
    out = np.zeros([N, C, out_h, out_w])
    max_indexs = []
    for sample_idx in range(N):
        sample = x[sample_idx]
        sample_max_value, sample_max_index = im2col(sample, 
                                                    pool_param['pool_height'], 
                                                    pool_param['pool_width'], 
                                                    pool_param['stride'])
        out[sample_idx] = sample_max_value
        max_indexs.append(sample_max_index)
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, max_indexs, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    # pass
    
    def im2col_backward(dout, max_index, out_h, out_w, C, HH, WW, stride):
        # dout [C, out_h, out_w]
        # max_index [out_h*out_w, C]
        # dout now [out_h*out_w, C]
        dout = dout.transpose(1, 2, 0).reshape([out_h*out_w, C])
        dout_reshape = np.zeros([out_h*out_w, C, HH*WW])
        for idx in range(out_h*out_w):
            for c_idx in range(C):
                dout_reshape[idx, c_idx, max_index[idx, c_idx]] = dout[idx, c_idx]
        dout_reshape = dout_reshape.reshape([out_h*out_w, C, HH, WW])
        #out [C, H, W]
        dx = np.zeros([C, H, W])
        for idx in range(out_h*out_w):
            h_idx = (idx // out_h) * stride
            w_idx = (idx % out_w) * stride
            dpatch = dout_reshape[idx]
            dx[:, h_idx:(h_idx+HH), w_idx:(w_idx+WW)] += dpatch
        
        return dx
    
    x, max_indexs, pool_param = cache
    N, C, H, W = x.shape
    out_h = (H - pool_param['pool_height']) // pool_param['stride'] + 1
    out_w = (W - pool_param['pool_width']) // pool_param['stride'] + 1
    dx = np.zeros(x.shape)
    for sample_idx in range(N):
        sample_max_idx = max_indexs[sample_idx]
        dout_sample = dout[sample_idx]
        dsample = im2col_backward(dout_sample, sample_max_idx, 
                                  out_h, out_w, C, 
                                  pool_param['pool_height'], pool_param['pool_width'], pool_param['stride'])
        dx[sample_idx] = dsample
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    # pass
    
    eps = bn_param.get('eps', 1e-5)
    
    N, C, H, W = x.shape
    x_reshape = x.transpose(0, 2, 3, 1).reshape([N*H*W, C])
    if bn_param['mode'] == 'train':
        out, cache = batchnorm_forward_alt(x_reshape, gamma, beta, bn_param)  
    elif bn_param['mode'] == 'test':
        out = (x_reshape - bn_param['running_mean']) / (np.sqrt(bn_param['running_var']) + eps) * gamma + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
    out = out.reshape([N, H, W, C]).transpose(0, 3, 1, 2)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    # pass
    N, C, H, W = dout.shape
    dout_reshape = dout.transpose(0, 2, 3, 1).reshape([N*H*W, C])
    dx, dgamma, dbeta = batchnorm_backward_alt(dout_reshape, cache)
    dx = dx.reshape([N, H, W, C]).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
