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
        if len(top) != 2:
            raise Exception("Need 2 outputs for probability distribution")

        self.beta = 0.95
        
        _, D, M, _ = bottom[0].data.shape
        self.xv, self.yv = np.meshgrid(np.arange(M), np.arange(M))
        self.channels = np.arange(D)
         
    def reshape(self, bottom, top):
        N, D, _, _ = bottom[0].data.shape
        top[0].reshape(*(N, 2 * D))
        top[1].reshape(*(N, D))

    def forward(self, bottom, top):
        N, D, M, _ = bottom[0].data.shape
        for k in xrange(len(bottom[0].data)):
            # Compute the expected 2D position for the probability distribution of each channel
            x_fc = np.sum(self.xv * bottom[0].data[k], axis=1)
            x_fc = np.sum(x_fc, axis=1)
            y_fc = np.sum(self.yv * bottom[0].data[k], axis=1)
            y_fc = np.sum(y_fc, axis=1)
            top[0].data[k][::2] = x_fc
            top[0].data[k][1::2] = y_fc

            # Save the s_cij for feature presence
            # TODO: Because feature points may fall between boundaries, sum probabilities of 3x3 window
            x_fc = np.floor(x_fc).astype(int)
            y_fc = np.floor(y_fc).astype(int)
            top[1].data[k] = (bottom[0].data[k][self.channels, x_fc, y_fc] > self.beta)

    def backward(self, top, propagate_down, bottom):
        # d x_fc/d_scij = i. self.xv
        # d y_fc/d_scij = j. self.yv
        d_xfc_scij, d_yfc_scij = self.xv, self.yv
        for k in xrange(len(bottom[0].data)):
            # top[0].diff[...] = d LOSS/d x_fc or d LOSS/d y_fc
            d_L_xfc = top[0].diff[k][::2]
            d_L_yfc = top[0].diff[k][1::2]
            for c in xrange(len(self.channels)):
                # TODO: RuntimeWarning: invalid value encountered in multiply (top[0].diff = inf). Issue caused by forward pass
                # d LOSS/d s_cij = d LOSS/d x_fc * d x_fc/d s_cij + d LOSS/d y_fc * d y_fc/d s_cij
                bottom[0].diff[k,c] = d_L_xfc[c] * d_xfc_scij + d_L_yfc[c] * d_yfc_scij
