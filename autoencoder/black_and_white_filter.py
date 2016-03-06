import caffe
import numpy as np


class BlackAndWhiteFilter(caffe.Layer):
    """
    Input: an embedding blob (shape N x M x M x D)
        N : items per batch
        M : dimensions of image
        D : number of filters
    Output: 1 top blob of shape N x M x M
    """
    
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Need 1 input for the black and white filter")
        if len(top) != 1:
            raise Exception("Need 1 output for the black and white filter")
         
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        # 0.21 R + 0.72 G + 0.07 B
        top[0].data = bottom[0].data[:,:,:,0] * 0.21 + bottom[0].data[:,:,:,1] * 0.72 + bottom[0].data[:,:,:,2] * 0.07

    def backward(self, top, propagate_down, bottom):
        pass
