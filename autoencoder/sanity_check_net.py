import argparse
import caffe
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run forward and backward pass")
    parser.add_argument("--proto", default="net.prototxt",
                        help="path to proto file")
    args = parser.parse_args()

    trained_net = caffe.Net(args.proto, caffe.TRAIN)

    print "Feature Points Dimensions:"
    print trained_net.blobs["spatialsoftmax"].data.shape

    print "running sanity check on forward and backward passes"
    trained_net.forward()
    trained_net.backward()
    print "loss:"
    print trained_net.blobs["loss"].data

    trained_net.forward()
    trained_net.backward()
    print "loss:"
    print trained_net.blobs["loss"].data
