import caffe
import numpy as np


class ProbabilityDistribution(caffe.Layer):
    """
    Input: a blob (shape N x D x M x M)
        N : items per batch
        D : number of filters
        M : dimensions of image
    Output: 1 top blob of shape N x 2 * D
    """
    
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Need 1 input for probability distribution")
        if len(top) != 1:
            raise Exception("Need 1 output for probability distribution")
         
    def reshape(self, bottom, top):
        N, D, _, _ = bottom[0].data.shape
        top[0].reshape(*(N, 2 * D))

    def forward(self, bottom, top):
        _, D, M, _ = bottom[0].data.shape
        
        xv, yv = np.meshgrid(np.arange(M), np.arange(M))

        for i in xrange(bottom[0].num):
            # Compute the expected 2D position for the probability distribution of each channel
            x_fc = np.sum(xv * bottom[0].data[i], axis=1)
            x_fc = np.sum(x_fc, axis=1)
            y_fc = np.sum(yv * bottom[0].data[i], axis=1)
            y_fc = np.sum(y_fc, axis=1)
            top[0].data[i][::2] = x_fc
            top[0].data[i][1::2] = y_fc

    def backward(self, top, propagate_down, bottom):
        # TODO: backprop alpha
        # top[0].diff[...] = d top / d y_fc or d top / d x_fc
        # d y_fc / d_scij = j
        # d x_fc / d_scij = i
        N, D, M, _ = bottom[0].data.shape
        x_deriv, y_deriv = np.meshgrid(np.arange(M), np.arange(M))
        for i in xrange(N):
            bottom[0].diff[i][::2] = x_deriv
            bottom[0].diff[i][1::2] = y_deriv
            # TODO: incorporate d top / d fc
