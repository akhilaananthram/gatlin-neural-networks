import caffe
import numpy as np


class SpatialSoftMax(caffe.Layer):
    """
    Input: an embedding blob (shape N x D x M x M)
        N : items per batch
        D : number of filters
        M : dimensions of image
    Output: 1 top blob of shape N x D x M x M
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
        N, D, M, _ = bottom[0].data.shape
        top[0].reshape(*(N, D, M, M))

    def forward(self, bottom, top):
        # for each image apply the exponential function to all channels
        exp_bottom = np.exp(bottom[0].data / self.alpha)

        for i in xrange(bottom[0].num):
            # axis=1 -> channels
            denominator = np.sum(exp_bottom[i], axis=1)
            denominator = np.sum(denominator, axis=1)
            top[0].data[i][:] = exp_bottom[i] / denominator[:, np.newaxis, np.newaxis]

    def backward(self, top, propagate_down, bottom):
        # TODO: backprop alpha
        # top[0].diff[...] = d top / d s_cij
        # d s_cij / d a_ci'j' = sc_ij (delta_ij - s_ci'j') where delta_ij = 1 if i=i', j=j', else 0
        pass
