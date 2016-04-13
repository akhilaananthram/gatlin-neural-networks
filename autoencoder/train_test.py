import argparse
import caffe
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run train and test")
    parser.add_argument("--solver", default="solver.prototxt",
                        help="Path to solver proto file")
    parser.add_argument("--model", default=None, type=str,
                        help="Path to pretrained caffe model")
    parser.add_argument("--niter", default=10, type=int,
                        help="Number of iterations for training")
    args = parser.parse_args()

    # Uncomment when training with a GPU
    #caffe.set_device(0)
    #caffe.set_mode_gpu()

    solver = caffe.SGDSolver(args.solver)

    # each output is (batch size, feature dim, spatial dim)
    print [(k, v.data.shape) for k, v in solver.net.blobs.items()]

    if args.model is not None:
        # Load pretrained weights
        solver.net.copy_from(args.model)

    train_loss = np.zeros(args.niter)
    for it in xrange(args.niter):
        print "Iteration {}".format(it)
        solver.step(1)  # SGD by Caffe

        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data
