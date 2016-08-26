import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class stn(object):
  """
  Implementation of Spatial transformer network:
  Consists of three functions which are basically sequential
  -- the localiser network takes input a feature map and op's
  (B,2,3) affine transform coordinates for every input in the
  batch.
  -- the grid generator accepts this transform coordinates and
  height and width and outputs a texture map which is then used
  -- by the bilinear interpolator which takes in the grid and
  input and produces a transformed image.
  """
  def __init__(filter_size=5, num_filters=128, hidden_dim=1024, dtype=np.flaot32):
    self.fsize = filter_size
    self.nfilters = num_filters
    self.hdim = hidden_dim
    self.dtype = dtype
    self.cache = {}

  def localiser_forward(inp):
    # input should be of the form (B,C,H,W)
    assert inp.ndim == 4
    N, C, H, W = inp.shape

    # I will assume the network is of the form
    # (conv-relu-pool)->fc_net->output and that conv layer has 128 filters and
    # outputs the dimensions same as the input image, then fc layer has 1024 
    # hidden units and finally output layer will be of the form (6,1,1) 
    # which I will reshape and return
    # Initialisation of the network

    assert C*H*W == 3072
    W1 = (2.0)/(C*H*W) * np.random.randn(self.nfilters, C, self.fsize, self.fsize).astype(self.dtype)
    b1 = np.zeros(self.nfilters)
    assert self.nfilters*(H/2)*(W/2) == 32768 
    W2 = (2.0)/(self.nfilters*(H/2)*(W/2)) * np.random.randn(self.hdim, self.nfilters, H/2, W/2).astype(self.dtype)
    b2 = np.zeros(self.hdim)
    W3 = (2.0/(self.hdim)) * np.random.randn(6, self.hdim, 1, 1).astype(self.dtype)
    b3 = np.zeros(6)
    
    # forward pass of the model
    conv_param = {'stride':1, 'pad':(self.fsize-1)/2}
    pool_param = {'pool_height':2, 'pool_width':2, 'stride':2}
    a1, c1 = conv_relu_pool_forward(inp, W1, b1, conv_param, pool_param)
    self.cache['c1'] = c1
    fc_param = {'stride':1, 'pad':0}
    a2, c2 = conv_relu_forward(a1, W2, b2, fc_param)
    self.cache['c2'] = c2
    a3, c3 = conv_forward_fast(a2, W3, b3, fc_param)
    self.cache['c3'] = c3
    a3 = a3.reshape(N, 2, 3)
    return a3, cache
  
  def localiser

  def grid_generator(tfmat, height, width):

