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
         
    def reshape(self, bottom, top):
        N, D, _, _ = bottom[0].data.shape
        top[0].reshape(*(N, 2 * D))
        top[1].reshape(*(N, D))

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

            # Save the s_cij for feature presence
            # TODO: Because feature points may fall between boundaries, sum probabilities of 3x3 window
            x_fc = np.floor(x_fc).astype(int)
            y_fc = np.floor(y_fc).astype(int)
            top[1].data[i] = (bottom[0].data[i][np.arange(D), x_fc, y_fc] > self.beta)

    def backward(self, top, propagate_down, bottom):
        # top[0].diff[...] = d LOSS/d x_fc or d LOSS/d y_fc
        # d x_fc/d_scij = i
        # d y_fc/d_scij = j
        N, D, M, _ = bottom[0].data.shape
        d_fcx_scij, d_fcy_scij = np.meshgrid(np.arange(M), np.arange(M))
        for k in xrange(N):
            # d LOSS/d s_cij = d LOSS/d x_fc * d x_fc/d s_cij + d LOSS/d y_fc * d y_fc/d s_cij
            d_L_fcx = top[0].diff[k][::2]
            d_L_fcy = top[0].diff[k][1::2]
            for c in xrange(D):
                # TODO: RuntimeWarning: invalid value encountered in multiply (top[0].diff = inf)
                bottom[0].diff[k,c] = d_L_fcx[c] * d_fcx_scij + d_L_fcy[c] * d_fcy_scij
