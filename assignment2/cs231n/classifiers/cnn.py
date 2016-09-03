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
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=64, filter_size=7,
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
    iDim = np.array(input_dim)
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.paramsarams.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.params['W1'] = (2.0/np.product(iDim)) * np.random.randn(num_filters, input_dim[0],filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['gamma1'] = np.ones(num_filters)
    self.params['beta1'] = np.zeros(num_filters)
    # Parameters for the parametric relu
    self.params['a1'] = np.full(num_filters, 0.25)

    # for the second convolution layer
    self.params['W2'] = (2.0/(num_filters*filter_size*filter_size)) * np.random.randn(num_filters, num_filters, filter_size, filter_size)
    self.params['b2'] = np.zeros(num_filters)
    self.params['gamma2'] = np.ones(num_filters)
    self.params['beta2'] = np.zeros(num_filters)
    self.params['a2'] = np.full(num_filters, 0.25)
 
    # for the fully connected layer
    self.params['W3'] = (2.0/(num_filters*(iDim[1]/4)*(iDim[1]/4))) * np.random.randn(hidden_dim, num_filters, input_dim[1]/4, input_dim[1]/4)
    self.params['b3'] = np.zeros(hidden_dim)
    self.params['gamma3'] = np.ones(hidden_dim)
    self.params['beta3'] = np.zeros(hidden_dim)
    self.params['a3'] = np.full(hidden_dim, 0.25)

    self.params['W4'] = (2.0/hidden_dim) * np.random.randn(num_classes, hidden_dim, 1,1)
    self.params['b4'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    N, C, H, W = X.shape
    W1, b1 = self.params['W1'], self.params['b1']
    gamma1, beta1 = self.params['gamma1'], self.params['beta1']
    ap1 = self.params['a1']
    W2, b2 = self.params['W2'], self.params['b2']
    gamma2, beta2 = self.params['gamma2'], self.params['beta2']
    ap2 = self.params['a2']
    W3, b3 = self.params['W3'], self.params['b3']
    gamma3, beta3 = self.params['gamma3'], self.params['beta3']
    ap3 = self.params['a3']
    W4, b4 = self.params['W4'], self.params['b4']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    fc_param = {'stride': 1, 'pad': 0}
    dropout_param = {'p': 0.5, 'mode': 'train'}
    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    # will have to define separate bn_params dictionary for each layer
    # since running mean and variance will be different for each layer
    bn_params1 = {'mode':'train'}
    bn_params2 = {'mode':'train'}
    bn_params3 = {'mode':'train'}

    # defining helper function and enclosing conv_bn_relu in one
    def conv_bn_relu_f(X, W, b, gamma, beta, a, conv_param, bn_param):
      ca1, cc1 = conv_forward_fast(X, W, b, conv_param)
      ca2, cc2 = spatial_batchnorm_forward(ca1, gamma, beta, bn_param)
      out1, cc3 = parametric_relu_forward(ca2, a)
      ccache = (cc1, cc2, cc3)
      return out1, ccache

    def conv_bn_relu_b(cdout, ccache):
      cc1, cc2, cc3 = ccache
      cda3, cda = parametric_relu_backward(cdout, cc3)
      cda2, cdgamma, cdbeta = spatial_batchnorm_backward(cda3, cc2)
      cda1, cdW, cdb = conv_backward_fast(cda2, cc1)
      return cda1, cdW, cdb, cdgamma, cdbeta, cda

    scores = None
    N = X.shape[0]
    num_classes = W4.shape[0]
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    a1, c1 = conv_bn_relu_f(X, W1, b1, gamma1, beta1, ap1, conv_param, bn_params1)
    a2, c2 = max_pool_forward_fast(a1, pool_param)

    # second layer convolution and max pool
    a3, c3 = conv_bn_relu_f(a2, W2, b2, gamma2, beta2, ap2, conv_param, bn_params2)
    a4, c4 = max_pool_forward_fast(a3, pool_param)

    # for the fully connected layer
    a5, c5 = conv_bn_relu_f(a4, W3, b3, gamma3, beta3, ap3, fc_param, bn_params3)

    # for the output layer
    a6, c6 = conv_forward_fast(a5, W4, b4, fc_param)
    a6 = a6.reshape(N, num_classes)
    
    scores = a6
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
    # for self.paramsarams[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, d_softmax = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3)+ np.sum(W4*W4))
    d_softmax = d_softmax.reshape(N,num_classes,1,1)
    da6, dw4, db4 = conv_backward_fast(d_softmax, c6)
    # if y == None:
    #   dropout_param['mode'] = 'test'
    # dd1 = dropout_backward(da4, cd1)
    da5, dw3, db3, dgamma3, dbeta3, da_p3 = conv_bn_relu_b(da6, c5)
    da4 = max_pool_backward_fast(da5, c4)
    da3, dw2, db2, dgamma2, dbeta2, da_p2 = conv_bn_relu_b(da4, c3)
    da2 = max_pool_backward_fast(da3, c2)
    da1, dw1, db1, dgamma1, dbeta1, da_p1 = conv_bn_relu_b(da2, c1)

    grads['W4'] = dw4 + self.reg*W4
    grads['b4'] = db4
    grads['W3'] = dw3 + self.reg*W3
    grads['b3'] = db3
    grads['W2'] = dw2 + self.reg*W2
    grads['b2'] = db2
    grads['W1'] = dw1 + self.reg*W1
    grads['b1'] = db1
    grads['gamma1'] = dgamma1
    grads['gamma2'] = dgamma2
    grads['gamma3'] = dgamma3
    grads['beta1'] = dbeta1
    grads['beta2'] = dbeta2
    grads['beta3'] = dbeta3
    grads['a1'] = da_p1
    grads['a2'] = da_p2
    grads['a3'] = da_p3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads