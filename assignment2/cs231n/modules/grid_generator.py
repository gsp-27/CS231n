import numpy as np

def grid_generator_forward(tf_mat, height, width):
  assert height > 1
  assert width > 1
  
  print height, width
  batch_size = tf_mat.shape[0]
  # since the input transform matrix is of size 2,3 and we want
  # to create a texture map out of it, it will require 3 dims.
  basegrid = np.ndarray(shape=(height, width, 3))

  # normalise the pixel values as mentioned in the paper.
  # the values are [xi, yi, 1]
  for i in range(1,height+1):
    basegrid[i-1, :, 0] = -1 + ((i-1)/(float(height)-1))*2

  for i in range(1,width+1):
    basegrid[:, i-1, 1] = (-1 + ((i-1)/(float(width)-1))*2)

  basegrid[:, :, 2] = 1
  batchGrid = np.ndarray((batch_size, height, width, 3))
  batchGrid[:] = basegrid
  print batchGrid

  # now that i have got the grid with normalised pixel coordinates,
  # I need to perform actual transformation by multiplying the grid
  # with the transformation matrix. Output should be of the form 
  # (B, Height, Width, 2)
  flattened_batch = batchGrid.reshape(batch_size, height*width, 3)
  ttf_mat = np.transpose(tf_mat, (0,2,1))
  output_flattened = np.ndarray((batch_size, height*width, 2))
  for i in range(batch_size):
    current_ip = flattened_batch[i]
    current_tf = ttf_mat[i]
    assert current_ip.shape == (25,3)
    assert current_tf.shape == (3,2)
    output_flattened[i] = np.dot(current_ip, current_tf)
  return output_flattened, flattened_batch

def grid_generator_backward(dout, batchGrid):
  batch_size = dout.shape[0]
  da = np.ndarray((batch_size, dout.shape[2], batchGrid.shape[2]))
  assert da.shape == (batch_size, 2, 3)
  for i in range(batch_size):
    current_dout = dout[i]
    current_batch = batchGrid[i]
    out = np.dot(current_batch.T, current_dout)
    da[i] = out.T
  assert da.shape == (batch_size, 2, 3)
  return da
