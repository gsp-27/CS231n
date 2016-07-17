import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    W1 = weight_scale * np.random.randn(input_dim*hidden_dim).reshape(input_dim, hidden_dim)
    b1 = np.zeros(hidden_dim)
    
    self.params['W1'] = W1
    self.params['b1'] = b1

    # Second affine layer will have 100 inputs from previous layer and it will produce 10 outputs with softmax non linearity
    W2 = weight_scale * np.random.randn(hidden_dim*num_classes).reshape(hidden_dim, num_classes)
    b2 = np.zeros(num_classes)

    self.params['W2'] = W2
    self.params['b2'] = b2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    W1 = self.params['W1']
    b1 = self.params['b1']
    out1, cache1 = affine_relu_forward(X, W1, b1)
    
    W2 = self.params['W2']
    b2 = self.params['b2']
    out2, cache2 = affine_forward(out1, W2, b2)
    scores = out2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
      
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
    loss, d_softmax = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
    d_affine2, dw2, db2 = affine_backward(d_softmax, cache2)
    dx, dw1, db1 = affine_relu_backward(d_affine2, cache1)
    grads['W2'] = dw2 + self.reg * W2
    grads['b2'] = db2
    grads['W1'] = dw1 + self.reg * W1
    grads['b1'] = db1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """
    
  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    hidden_dims.append(num_classes)

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    for i in range(self.num_layers):
      weight_init = weight_scale * np.random.randn(input_dim * hidden_dims[i]).reshape(input_dim, hidden_dims[i])
      bias_init = np.zeros(hidden_dims[i])
      self.params["W" + str(i+1)] = weight_init
      self.params["b" + str(i+1)] = bias_init
      input_dim = hidden_dims[i]

    if self.use_batchnorm:
      for i in xrange(1,self.num_layers):
        gamma_init = np.ones(hidden_dims[i-1])
        beta_init = np.zeros(hidden_dims[i-1])
        self.params["gamma"+str(i)] = gamma_init
        self.params["beta"+str(i)] = beta_init

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
        
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
      
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
      
  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    def affine_bn_relu_f(x, w, b, beta, gamma, bn_params):
      a, fc_cache = affine_forward(x, w, b)
      b, bn_cache = batchnorm_forward(a, gamma, beta, bn_params)
      out, relu_cache = relu_forward(b)
      cache = (fc_cache, bn_cache, relu_cache)
      return out, cache

    def affine_bn_relu_b(dout, cache):
      """
      Backward pass for the affine-relu convenience layer
      """
      fc_cache, bn_cache, relu_cache = cache
      da = relu_backward(dout, relu_cache)
      dbn, dgamma, dbeta = batchnorm_backward(da, bn_cache)
      dx, dw, db = affine_backward(dbn, fc_cache)
      return dx, dw, db, dgamma, dbeta
      
    N = X.shape[0]
    num_layers = self.num_layers
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
      if self.use_batchnorm:
        for bn_param in self.bn_params:
          bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    # we have three things here, do something for input then same thing but for 
    # hidden layers and finally affine_forward pass for output layer
    # for input
    a, z, dropout_cache = list(), list(), list()
    a.append(X)
    # for batchnorm i will have to update this to account for the fact that there is a batch norm layer before
    # each relu layer, and each batchnorm layer will take affine forward as input and output will go to relu in.
    for i in range(num_layers - 1):
      if self.use_batchnorm: 
        _act, _cache = affine_bn_relu_f(a[i], self.params['W'+str(i+1)], self.params['b'+str(i+1)], self.params['beta'+str(i+1)], self.params['gamma'+str(i+1)], self.bn_params[i])
        if self.use_dropout:
          _act, _cache1 = dropout_forward(_act, self.dropout_param)
          dropout_cache.append(_cache1)
        a.append(_act)
        z.append(_cache)

      else:
        _act, _cache = affine_relu_forward(a[i], self.params["W"+str(i+1)], self.params['b'+str(i+1)])
        if self.use_dropout:
          _act, _cache1 = dropout_forward(_act, self.dropout_param)
          dropout_cache.append(_cache1)
        a.append(_act)
        z.append(_cache)

    # for output layer
    _act, _cache = affine_forward(a[num_layers - 1], self.params['W'+str(num_layers)], self.params['b'+str(num_layers)])
    a.append(_act)
    z.append(_cache)
    scores = a[-1]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, d_softmax = softmax_loss(scores, y)
    regularization = 0
    for i in range(1, num_layers+1):
      regularization += np.sum(self.params['W' + str(i)] * self.params['W' + str(i)])
      loss += 0.5 * self.reg * regularization

    # print len(z), " ", len(a)
    # Computing the gradients, I have gradient for the softmax layer I can use it
    # to compute the gradients of the affine output layer and the repeat the process
    # d_affine2, dw2, db2 = affine_backward(d_softmax, cache2)
    # grads['W2'] = dw2 + self.reg * W2
    da = [0]*(num_layers+1)
    dw = [0]*(num_layers+1)
    db = [0]*(num_layers+1)
    if self.use_batchnorm:
      dgamma = [0]*(num_layers)
      dbeta = [0]*(num_layers)

    # first the output layer
    da[-1], dw[-1], db[-1] = affine_backward(d_softmax, z[-1])
    grads['W'+str(num_layers)] = dw[-1] + self.reg * self.params['W'+str(num_layers)]
    grads['b'+str(num_layers)] = db[-1]
    for i in range(num_layers-1, 0, -1):
      if self.use_dropout:
        da[i+1] = dropout_backward(da[i+1], dropout_cache[i-1])
      if self.use_batchnorm:
        da[i], dw[i], db[i], dgamma[i], dbeta[i] = affine_bn_relu_b(da[i+1], z[i-1])
      else:
        da[i], dw[i], db[i] = affine_relu_backward(da[i+1], z[i-1])

        grads['W'+str(i)] = dw[i] + self.reg * self.params['W'+str(i)]
        grads['b'+str(i)] = db[i]
        if self.use_batchnorm:
          grads['gamma'+str(i)] = dgamma[i]
          grads['beta'+str(i)] = dbeta[i]
          ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
