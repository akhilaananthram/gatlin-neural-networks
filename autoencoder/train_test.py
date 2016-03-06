import argparse
import caffe
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run train and test")
    parser.add_argument("--solver", default="sgd_solver.prototxt",
                        help="Path to quick solver proto file")
    parser.add_argument("--model", default=None, type=str,
                        help="Path to pretrained caffe model")
    args = parser.parser_args()

    # Uncomment when training with a GPU
    #caffe.set_device(0)
    #caffe.set_mode_gpu()

    solver = caffe.SGDSolver(args.solver)

    if args.model is not None:
        # Load pretrained weights
        solver.net.copy_from(args.model)

    solver.solve()
