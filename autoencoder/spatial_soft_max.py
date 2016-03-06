import caffe
import numpy as np


class SpatialSoftMax(caffe.Layer):
    """
    Input: an embedding blob (shape N x M x M x D)
        N : items per batch
        M : dimensions of image
        D : number of filters
    Output: 1 top blob of shape N x 2 * D
    """
    
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Need 1 input for the spatial soft max")
        if len(top) != 1:
            raise Exception("Need 1 output for the spatial soft max")

        # TODO: take as input
        self.alpha = 0.1
         
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        _, M, _, _ = bottom.shape
        # for each image apply the softmax function to all channels
        exp_bottom = np.exp(bottom[0].data / self.alpha)
        
        xv, yv = np.meshgrid(np.arange(M), np.arange(M))

        for i in xrange(bottom[0].num):
            # axis=1 -> channels
            denominator = np.sum(exp_bottom[i], axis=1)
            softmax = exp_bottom[i] / denominator

            # Compute the expected 2D position for the probability distribution of each channel
            x_fc = np.sum(xv * softmax, axis=1)
            y_fc = np.sum(yv * softmax, axis=1)
            top[0].data[i][::2] = x_fc
            top[0].data[i][1::2] = y_fc

    def backward(self, top, propagate_down, bottom):
        # TODO: backprop alpha
        pass
