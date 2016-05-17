import argparse
import caffe
import os
import sys
import caffetools

from caffe import layers as L
from caffe import params as P
from constants import *

def add_data_layer(n, data, batch_size, phase, data_type, mean_file, blob_name="original"):
    # TODO: divide by STD
    # subtract the mean
    transform_param = {}
    if mean_file is not None:
        transform_param["mean_file"] = mean_file

    # labels are not used because this is an autoencoder
    if data_type == "txt":
        n[blob_name], n.labels = L.ImageData(source=data, batch_size=batch_size, ntop=2,
                                             new_height=SIZE, new_width=SIZE, shuffle=True,
                                             include={"phase": phase}, mirror=(not phase),
                                             transform_param=transform_param)
    elif data_type == "db":
        n[blob_name] = L.Data(source=data, batch_size=batch_size,
                              backend=P.Data.LMDB, mirror=(not phase),
                              include={"phase": phase},
                              transform_param=transform_param)
    else:
        n[blob_name] = L.DummyData(shape={"dim":[batch_size, 3, SIZE, SIZE]},
                                   dummy_data_param={"data_filler": {"type": "xavier"}},
                                   ntop=1)

def add_intermediate_layers(n):
    """
    Net based on paper
    """
    # Want weights to be smaller if more inputs
    n.conv1 = L.Convolution(n.original, num_output=64, kernel_size=7, stride=2,
                            weight_filler={"type": "xavier"},
                            #weight_filler={"type": "gaussian", "std": 0.01},
                            bias_filler={"type": "constant"})
    n.bn1 = L.BatchNorm(n.conv1)
    n.relu1 = L.ReLU(n.bn1)
    n.conv2 = L.Convolution(n.relu1, num_output=32, kernel_size=5,
                            weight_filler={"type": "xavier"},
                            #weight_filler={"type": "gaussian", "std": 0.01},
                            bias_filler={"type": "constant"})
    n.bn2 = L.BatchNorm(n.conv2)
    n.relu2 = L.ReLU(n.bn2)
    n.conv3 = L.Convolution(n.relu2, num_output=16, kernel_size=5,
                            weight_filler={"type": "xavier"},
                            #weight_filler={"type": "gaussian", "std": 0.01},
                            bias_filler={"type": "constant"})
    n.bn3 = L.BatchNorm(n.conv3)
    n.relu3 = L.ReLU(n.bn3)

    n.softmax = L.Python(n.relu3, name="softmax", ntop=1,
                                python_param={"module": "soft_max",
                                              "layer": "SoftMax"})
    n.probabilitydist, n.s_cij = L.Python(n.softmax, name="probabilitydist", ntop=2,
                                          python_param={"module": "probability_distribution",
                                                        "layer": "ProbabilityDistribution"})

def gps_net(images, batch_size, phase, data_type, mean_file):
    n = caffe.NetSpec()

    add_data_layer(n, images, batch_size, phase, data_type, mean_file)

    # Get black and white
    n.blackandwhite = L.Python(n.original, name="blackandwhite", ntop=1,
                                python_param={"module": "black_and_white_filter",
                                              "layer": "BlackAndWhiteFilter"})
    # Downsample with max pooling
    # w_out = math.floor((w_in + 2 * p - k) / s) + 1
    n.downsample = L.Pooling(n.blackandwhite, kernel_size=4, stride=4)
    n.downsample_flat = L.Flatten(n.downsample)

    add_intermediate_layers(n)

    n.reconstruction = L.InnerProduct(n.probabilitydist, num_output=DOWNSAMPLE * DOWNSAMPLE,
                                      weight_filler={"type": "xavier"},
                                      bias_filler={"type": "constant"})
    # TODO: add smoothness penalty as another loss function
    n.loss = L.EuclideanLoss(n.reconstruction, n.downsample_flat)

    return str(n.to_proto())

def hsv_net(images, batch_size, phase, data_type, mean_file, hsv_file):
    n = caffe.NetSpec()

    add_data_layer(n, images, batch_size, phase, data_type, mean_file)
    add_data_layer(n, hsv_file, batch_size, phase, data_type, mean_file, "hsv_encoding")

    add_intermediate_layers(n)

    n.loss = L.EuclideanLoss(n.probabilitydist, n.hsv_encoding)

    return str(n.to_proto())

def get_data_type(path):
    if os.path.isfile(path):
        return "txt"
    elif os.path.isdir(path):
        return "db"
    else:
        return "dummy"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="setup net from database or txt file")
    parser.add_argument("--train", default="", help="path to database directory or txt file of paths")
    parser.add_argument("--val", default="", help="path to database directory or txt file of paths")
    parser.add_argument("--train-proto", default="train.prototxt", help="path to train proto file")
    parser.add_argument("--val-proto", default="val.prototxt", help="path to val proto file")
    parser.add_argument("--solver", default="solver.prototxt", help="path to solver proto file")
    parser.add_argument("--mean", default=None, help="path to mean binary proto file")
    args = parser.parse_args()

    with open(args.train_proto, "w") as f:
        f.write(gps_net(args.train, BATCH_TRAIN, 0, get_data_type(args.train), args.mean))
    with open(args.val_proto, "w") as f:
        f.write(gps_net(args.val, BATCH_TEST, 1, get_data_type(args.val), args.mean))

    solver = caffetools.CaffeSolver(net_prototxt_path=args.train_proto, testnet_prototxt_path=args.val_proto)
    solver.sp["test_iter"] = "1000"
    solver.sp["test_interval"] = "5000"
    solver.sp["test_initialization"] = "false"
    solver.sp["display"] = "40"
    solver.sp["average_loss"] = "40"
    solver.sp["base_lr"] = "0.001"
    solver.sp["lr_policy"] = '"exp"'
    solver.sp["max_iter"] = "350000"
    solver.sp["weight_decay"] = "0.0002"
    solver.sp["gamma"] = "0.9999"
    solver.sp["power"] = "0.00002"
    solver.sp["snapshot_prefix"] = '"encodings/snapshots"'
    solver.sp["solver_mode"] = "CPU"
    solver.write(args.solver)
