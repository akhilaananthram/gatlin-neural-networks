import argparse
import atexit
import caffe
import numpy as np
import warnings
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run train and test")
    parser.add_argument("--solver", default="solver.prototxt",
                        help="Path to solver proto file")
    parser.add_argument("--model", default=None, type=str,
                        help="Path to pretrained caffe model")
    parser.add_argument("--niter", default=350000, type=int,
                        help="Number of iterations for training")
    parser.add_argument("--blob-file", default=None, type=str,
                        help="Path to blob values")
    parser.add_argument("--train", dest="train", default=None, type=str,
                        help="Path to train file")
    args = parser.parse_args()

    # Uncomment when training with a GPU
    #caffe.set_device(0)
    #caffe.set_mode_gpu()

    # Debug
    #np.seterr(all='warn')
    #warnings.filterwarnings('error')

    # Use Adam to avoid worrying about the magnitude of the loss
    #solver = caffe.SGDSolver(args.solver)
    solver = caffe.AdamSolver(args.solver)
    
    # TODO: load from snapshot

    # each output is (batch size, feature dim, spatial dim)
    print [(k, v.data.shape) for k, v in solver.net.blobs.items()]

    if args.model is not None:
        # Load pretrained weights
        solver.net.copy_from(args.model)

    train_loss = np.zeros(args.niter)
    def save_loss():
        np.save("loss.npy", train_loss)

    atexit.register(save_loss)

    blobs = {}
    if args.blob_file is not None:
        with open(args.blob_file) as f:
            for line in f:
                label, xys = line.strip().split(",")
                blobs[label] = [float(x) for x in xys.split(" ")]

    # TODO: iterate over train file to get order of images
    train_labels = []
    if args.train is not None:
        with open(args.train) as f:
            for line in f:
                line = line.strip()
                filename, _ = line.split(" ")
                label, _ = os.path.splitext(os.path.basename(filename))
                train_labels.append(label)

    for it in xrange(args.niter):
        if it % 1000 == 0:
            print "Iteration {}".format(it)
        batch_blobs = []
        for l in train_labels[it * 64: (it + 1) * 64]:
            if l in blobs:
                batch_blobs.append(blobs[l])

        if len(batch_blobs) > 0:
            solver.net.blobs["hsv_encoding"].data[...] = np.array(batch_blobs)

        solver.step(1)  # SGD by Caffe

        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data

    print "Finished train - test"
