import argparse
import caffe
import tools


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
    black_and_white = trained_net.blobs["blackandwhite"].data
    probabilitydist = trained_net.blobs["probabilitydist"].data

    for i in xrange(len(trained_net.blobs["probabilitydist"].data)):
        tools.plot_reconstruction(reconstruction[i], "reconstructions/{}.png".format(i))
        # Remove axis for channels
        tools.plot_features(black_and_white[i][0], probabilitydist[i], "reconstructions/{}f.png".format(i))
