import caffe
import os
import sys

from caffe import layers as L
from caffe import params as P

BATCH = 64  # batch size
SIZE = 240 # size of image for input
DOWNSAMPLE = 60 # size of downsampled black and white image

def pynet(training_images, batch_size):
    n = caffe.NetSpec()
    
    # labels are not used because this is an autoencoder
    #n.original = L.ImageData(source=training_images, batch_size=batch_size,
    #                          new_height=SIZE, new_width=SIZE, shuffle=True)
    # TODO: subtract the mean
    n.original = L.Data(source=training_images, batch_size=batch_size,
                        crop_size=SIZE, backend=P.Data.LMDB, mirror=1)
    # Get black and white
    n.blackandwhite = L.Python(n.original, name="blackandwhite", ntop=1,
                                python_param={"module": "black_and_white_filter",
                                              "layer": "BlackAndWhiteFilter"})
    # Downsample with max pooling
    # w_out = math.floor((w_in + 2 * p - k) / s) + 1
    n.downsample = L.Pooling(n.blackandwhite, kernel_size=4, stride=4)
    n.downsample_flat = L.Flatten(n.downsample)

    # TODO: initialize weights from pretrained network
    # Want weights to be smaller if more inputs
    n.conv1 = L.Convolution(n.original, num_output=64, kernel_size=7, stride=2,
                            weight_filler={"type": "gaussian", "std": 0.01},
                            #weight_filler={"type": "xavier"},
                            bias_filler={"type": "constant"})
    n.bn1 = L.BatchNorm(n.conv1)
    n.relu1 = L.ReLU(n.bn1)
    n.conv2 = L.Convolution(n.relu1, num_output=32, kernel_size=5,
                            weight_filler={"type": "gaussian", "std": 0.01},
                            #weight_filler={"type": "xavier"},
                            bias_filler={"type": "constant"})
    n.bn2 = L.BatchNorm(n.conv2)
    n.relu2 = L.ReLU(n.bn2)
    n.conv3 = L.Convolution(n.relu2, num_output=16, kernel_size=5,
                            weight_filler={"type": "gaussian", "std": 0.01},
                            #weight_filler={"type": "xavier"},
                            bias_filler={"type": "constant"})
    n.bn3 = L.BatchNorm(n.conv3)
    n.relu3 = L.ReLU(n.bn3)
    # Compute the softmax for each channel in the blob
    n.softmax = L.Python(n.relu3, name="softmax", ntop=1,
                                python_param={"module": "soft_max",
                                              "layer": "SoftMax"})
    n.probabilitydist, n.s_cij = L.Python(n.softmax, name="probabilitydist", ntop=2,
                                          python_param={"module": "probability_distribution",
                                                        "layer": "ProbabilityDistribution"})

    n.reconstruction = L.InnerProduct(n.probabilitydist, num_output=DOWNSAMPLE * DOWNSAMPLE,
                                      weight_filler={"type": "gaussian", "std": 0.01},
                                      #weight_filler={"type": "xavier"},
                                      bias_filler={"type": "constant"})
    # TODO: add smoothness penalty
    n.loss = L.EuclideanLoss(n.reconstruction, n.downsample_flat)

    return str(n.to_proto())


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.realpath(__file__))  

    with open(os.path.join(curr_dir, "net.prototxt"), "w") as f:
        f.write(pynet(os.path.join(curr_dir, "img_db"), BATCH))

