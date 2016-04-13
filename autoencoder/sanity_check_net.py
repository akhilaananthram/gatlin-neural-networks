import argparse
import caffe
import math
import numpy as np

from PIL import Image # Pillow

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run forward and backward pass")
    parser.add_argument("--proto", default="train.prototxt",
                        help="path to proto file")
    args = parser.parse_args()

    trained_net = caffe.Net(args.proto, caffe.TRAIN)

    print "running sanity check on forward and backward passes"
    trained_net.forward()
    trained_net.backward()
    print "loss:"
    print trained_net.blobs["loss"].data

    trained_net.forward()
    trained_net.backward()
    print "loss:"
    print trained_net.blobs["loss"].data

    print "Feature Points Dimensions:"
    print trained_net.blobs["probabilitydist"].data.shape

    reconstruction = trained_net.blobs["reconstruction"].data
    N, MM = reconstruction.shape
    reconstruction = np.reshape(reconstruction, (N, math.sqrt(MM), math.sqrt(MM)))
    img = Image.fromarray(reconstruction[0])
    img = img.convert('RGB')
    img.save('reconstruction.png')
