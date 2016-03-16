import caffe
import numpy as np


class SoftMax(caffe.Layer):
    """
    Input: an embedding blob (shape N x D x M x M)
        N : items per batch
        D : number of filters
        M : dimensions of image
    Output: 1 top blob of shape N x D x M x M
    alpha = 1 is basic softmax
    """
    
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Need 1 input for the spatial soft max")
        if len(top) != 1:
            raise Exception("Need 1 output for the spatial soft max")

        # TODO: take as input
        self.alpha = 1
         
    def reshape(self, bottom, top):
        N, D, M, _ = bottom[0].data.shape
        top[0].reshape(*(N, D, M, M))

    def forward(self, bottom, top):
        for i in xrange(bottom[0].num):
            # axis=1 -> channels
            # subtract the max to avoid numerical issues
            largest = np.max(np.max(bottom[0].data[i], axis=1), axis=1)
            # for each image apply the exponential function to all channels
            exp_bottom = np.exp((bottom[0].data[i] - largest[:, np.newaxis, np.newaxis]) / self.alpha)

            denominator = np.sum(np.sum(exp_bottom, axis=1), axis=1)
            top[0].data[i][:] = exp_bottom / denominator[:, np.newaxis, np.newaxis]

    def backward(self, top, propagate_down, bottom):
        # top[0].diff[...] = d top / d s_cij
        N, D, M, _ = bottom[0].data.shape
        delta_ij = np.eye(M)
        for k in xrange(N):
            # d s_cij / d a_ci'j' = sc_ij (delta_ij - s_ci'j') where delta_ij = 1 if i=i', j=j', else 0
            for c in xrange(D):
                d_scij_acij = top[0].data[k,c] * (delta_ij - top[0].data[k,c])
                bottom[0].diff[k,c] = np.dot(top[0].data[k,c], d_scij_acij)

        # TODO: backprop alpha
