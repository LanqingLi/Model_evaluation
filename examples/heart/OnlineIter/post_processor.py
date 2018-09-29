import os
import numpy as np
import mxnet as mx
import sys

sys.path.append("..")

def get_calcium_mask(img_array, binary_map, calcium_thresh):
    assert img_array.shape == binary_map.shape, 'image array and binary map must have the same shape'
    assert np.prod(binary_map.shape) == np.count_nonzero(binary_map == 0) + np.count_nonzero(binary_map == 1), \
        'binary map must only contain 0 or 1'
    binary_map_cpy = binary_map.copy()
    binary_map_cpy[img_array < calcium_thresh] = 0
    return binary_map_cpy
