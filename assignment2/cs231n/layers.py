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
  N = x.shape[0]
  D = np.prod(x.shape[1:])
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  x_reshaped = x.reshape(N, D)
  out = x_reshaped.dot(w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
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
  N = x.shape[0]
  D = np.prod(x.shape[1:])
  x_reshaped = x.reshape(N, D)
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  dx = dout.dot(w.T)
  dw = x_reshaped.T.dot(dout)
  db = np.sum(dout, axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  dx = dx.reshape(*x.shape)
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
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  gzero_ind = x >= 0.0
  out = gzero_ind * x
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
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
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  calc_idx = x > 0
  dx = calc_idx * dout
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

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

  out, cache = None, dict()
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    _mean = np.mean(x, axis=0)
    _var = np.var(x, axis=0)
    assert "problem with mean computation", _mean.shape == D
    assert "problem with variance computation", _var.shape == D
    x_hat = x - _mean
    x_hat = x_hat / (np.sqrt(_var + eps))
    assert "dimensions of x should be N x D", x_hat.shape == (N, D)
    out = gamma * x_hat + beta
    running_mean = momentum * running_mean + (1- momentum) * _mean
    running_var = momentum * running_var + (1 - momentum) * _var
    cache['gamma'] = gamma
    cache['x_hat'] = x_hat
    cache['mean'] = _mean
    cache['var'] = _var
    cache['eps'] = eps
    cache['x'] = x
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    x_hat = x - running_mean
    x_hat = x_hat / (np.sqrt(running_var + eps))
    out = gamma * x_hat + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
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
  N, D = np.shape(dout)
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  dx_hat = dout * cache['gamma'] # output should be NxD
  assert "dimension mismatch x_hat",dx_hat.shape == (N, D)
  
  t1 = cache['x'] - cache['mean']
  t2 = (-0.5)*((cache['var'] + cache['eps'])**(-1.5))
  t1 = t1 * t2
  d_var = np.sum(dx_hat * t1, axis=0)
  assert "dimensional mismatch variance",d_var.shape == D

  tmean1 = (-1)*((cache['var'] + cache['eps'])**(-0.5))
  d_mean = np.sum(dx_hat * tmean1, axis=0)
  assert "dimension mismatch mean", d_mean.shape == D

  tmean1 = (-1)*tmean1
  tx1 =   dx_hat * tmean1
  tx2 = d_mean * (1.0 / float(N))
  tx3 = d_var * (2 * (cache['x'] - cache['mean']) / N)
  dx = tx1 + tx2 + tx3
  assert "dimension mismatch x", dx.shape == (N, D)

  dgamma = np.sum(dout * cache['x_hat'], axis=0)
  dbeta = np.sum(dout, axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


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
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
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
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  """ 
  What is dropout: Dropout randomly samples the whole neural network during training
  to prevent overfitting and give better results during test time. Neural nets are 
  generally used in application where there exists complex relationships between input
  and output, if the capacity of the network is high enough then there is chance that
  we will get 100% accuracy during training, but this network will surely give poor 
  results during test time. To prevent this co-adaption of hidden neurons on training
  data, we randomnly on average drop half of the units from the network, with probability
  p, which is a hyperparameter but is generally set to 0.5. Alternative view is these
  sampled networks can be considered as ensemble of many neural net architectures who all
  share weights among them. And is generally observed ensemble models give better results
  and are immune to overfitting.
  """
  
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    # first i create a mask,keep only those whose probability is greater than p
    mask = (np.random.rand(*x.shape) < p) / p
    out = x * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

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
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

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
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  pad = conv_param['pad']
  stride = conv_param['stride']
  # getting the image and filter dimension
  C, H, W = x[0].shape
  C, HH, WW = w[0].shape

  # calculate the width and height of the output blob
  oH = 1 + (H + 2*pad - HH) / stride
  oW = 1 + (W + 2*pad - WW) / stride
  # the output array size
  out = np.ndarray((x.shape[0], w.shape[0], oH, oW))
  v = np.ndarray((w.shape[0], oH, oW))

  # for each image, convolve it with all the filters
  for i in range(x.shape[0]):
    image = x[i]
    npad = ((0,0), (pad, pad), (pad, pad))
    image = np.pad(image, npad, mode='constant', constant_values=0)
    for f in range(w.shape[0]):
      kernel = w[f]
      
      for ii in range(oH):
        hSt, hEnd = ii*stride, (ii*stride+HH)
        for jj in range(oW):
          wSt, wEnd = jj*stride, (jj*stride+WW)
          patch = image[:, hSt:hEnd, wSt:wEnd]
          v[f][ii][jj] = np.sum(patch * kernel) + b[f]
    # Here i get a single image convolved with all the filters
    out[i] = v
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
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
  x, w, b, conv_param = cache
  stride = conv_param['stride']
  pad = conv_param['pad']
  HH, WW = w[0][0].shape
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)
  N, C, H, W = dout.shape
  for i in range(N):
    # get the x and the error and pad the x
    dim = dout[i]
    im = x[i]
    dxim = dx[i] # 3x5x5
    npad = ((0,0), (pad, pad), (pad, pad))
    im = np.pad(im, npad, mode='constant', constant_values=0)
    dxim = np.pad(dxim, npad, mode='constant', constant_values=0)
    # for each channel of the error, get the corresponding, flowing in error mat
    # note that this would be same for filter (w) and the dout but different for
    # the image
    for c in range(C):
      dim_blob = dim[c]
      kernel = w[c]
      
      # from the matrix get the value of err and also the image patch
      for ii in range(H):
        hst, hend = ii*stride, ii*stride+HH
        for jj in range(W):
          wst, wend = jj*stride, jj*stride+WW
          dw[c] += dim_blob[ii][jj]*im[:,hst:hend,wst:wend]
          db[c] += dim_blob[ii][jj]
          dxim[:, hst:hend, wst:wend] += dim_blob[ii][jj]*w[c]
    dx[i] = dxim[:, 1:-1, 1:-1]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
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
  ph_ = pool_param['pool_height']
  pw_ = pool_param['pool_width']
  stride_ = pool_param['stride']
  C, H, W = x[0].shape
  oh_ = 1 + (H-ph_)/stride_
  ow_ = 1 + (W-pw_)/stride_
  out = np.ndarray((x.shape[0], x.shape[1], oh_, ow_))
  max_idxs = list()
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  # get each image
  for i in range(x.shape[0]):
    im = x[i]

    # get each filtered image blob
    for c in range(im.shape[0]):
      im_blob = im[c]

      # for each patch of ph_*pw_ select the max in that area
      for ii in range(oh_):
        hst, hend = ii*stride_, (ii*stride_+ph_)
        for jj in range(ow_):
          wst, wend = jj*stride_, (jj*stride_+pw_)
          im_patch = im_blob[hst:hend, wst:wend]
          out[i][c][ii][jj] = np.max(im_patch)
          # im_patch is a square slice of the image and from this I am getting
          # max element, I use here argmax which will return me a single ele
          # array giving me the max index which i then append to the list
          a,b = np.unravel_index(im_patch.argmax(), im_patch.shape)
          max_idxs.append((a,b))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param, max_idxs)
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
  x, pool_param, max_idx = cache
  ph_ = pool_param['pool_height']
  pw_ = pool_param['pool_width']
  stride_ = pool_param['stride']
  dx = np.zeros_like(x)
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  # for each image
  for i in range(dout.shape[0]):
    image = dout[i]
    dx_image = dx[i]

    # for each channel
    for c in range(image.shape[0]):
      im_blob = image[c]
      dx_blob = dx_image[c]

      # we already have max index so index it
      # and then set that index in dx to dout[ii][jj]
      for ii in range(im_blob.shape[0]):
        hst, hend = ii*stride_, ii*stride_+ph_
        for jj in range(im_blob.shape[1]):
          wst, wend = jj*stride_, jj*stride_+pw_
          dx_patch = dx_blob[hst:hend, wst:wend]
          a,b = max_idx[0]
          max_idx = max_idx[1:]
          dx_patch[a][b]= im_blob[ii][jj]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
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

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N, C, H, W = x.shape
  x = np.transpose(x, [0,2,3,1])
  x = x.reshape(N*H*W, C)
  out_tmp, cache = batchnorm_forward(x, gamma, beta, bn_param)
  out_tmp = out_tmp.reshape(N, H, W, C)
  out = np.transpose(out_tmp, [0,3,1,2])
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

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

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N, C, H, W = dout.shape
  dout = np.transpose(dout, [0,2,3,1])
  dout = dout.reshape(N*H*W, C)
  dx_temp, dgamma, dbeta = batchnorm_backward(dout, cache)
  dx_temp = dx_temp.reshape(N, H, W, C)
  dx = np.transpose(dx_temp, [0,3,1,2])
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
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
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

def parametric_relu_forward(x, a):
  """
  Computes the parametric relu activation forward pass
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - a: parameters of shape (C,)

  Returns a tuple of:
  - activations: y = max(0, x_i) + a_i*min(0, x_i)
  - cache: two masks one for positive values and one for negative values
  """
  N,C,H,W = x.shape
  mask_gzero = x > 0
  assert mask_gzero.shape == x.shape
  mask_lzero = x <= 0
  assert mask_lzero.shape == x.shape
  temp_a = np.tile(a, N*H*W).reshape(N,C,H,W)
  assert temp_a.shape == x.shape
  assert (mask_lzero*x).shape == x.shape
  out = mask_gzero*x + temp_a*mask_lzero*x
  cache = (x, a)
  return out, cache

def parametric_relu_backward(dout, cache):
  """
  Computes the backward pass of the parametric relu activation
  as described in the paper "Delving deep into rectifiers" He et.al

  Inputs:
  - dout: upstream derivative
  - cache: (x, a)

  Outputs:
  - dx: of shape same as dout
  - da: of shape same as a
  """
  x, a = cache
  N, C, H, W = x.shape
  mask_gzero = x > 0
  mask_lzero = x <= 0
  temp_a = np.tile(a, N*H*W).reshape(N,C,H,W)
  dx = dout*mask_gzero + temp_a*dout*mask_lzero
  da = np.sum(dout*mask_lzero*x, axis=(0,2,3))
  return dx, da

def avg_pooling_forward(x, pool_params):
  """
    X: input of shape (N, C, H, W)
    pool_params : a dictionary containing filter width, filter height,
            and stride
  """
  N, C, H, W = x.shape
  ph_ = pool_params['pool_height']
  pw_ = pool_params['pool_width']
  stride = pool_params['stride']

  # compute the output height and width
  _oh = (H - ph_) / stride + 1
  _ow = (W - pw_) / stride + 1

  # declare an output tensor which will hold values
  out = np.zeros((N, C, _oh, _ow), dtype=np.float)

  # fill the array with average pooling
  for i in range(N):
    for c in range(C):
      for ii in range(_oh):
        hst, hend = ii*stride, ii*stride+ph_
        for jj in range(_ow):
          wst, wend = jj*stride, jj*stride+pw_
          # get the image patch
          im_patch = x[i, c, hst:hend, wst:wend]
          # sum this patch divide by ph*pw and assign it to output
          out[i,c,ii,jj] = np.sum(im_patch) / (ph_*pw_)

  return out, pool_params

def avg_pooling_backward(dout, cache):
  """
    dout: numpy array of size(N, C, H, W)
  """
  # compute the size of the input tensor
  N, C, _oh, _ow = dout.shape
  ph_ = cache['pool_height']
  pw_ = cache['pool_width']
  stride = cache['stride']

  H = (_oh - 1)*stride + ph_
  W = (_ow - 1)*stride + pw_
  dx = np.zeros((N, C, H, W), dtype=np.float)

  for i in range(N):
    for c in range(C):
      for ii in range(_oh):
        hst, hend = ii*stride, ii*stride+ph_
        for jj in range(_ow):
          wst, wend = jj*stride, jj*stride+pw_
          dx[i,c,hst:hend, wst:wend] = dout[i,c,ii,jj]/(ph_*pw_)

  return dx