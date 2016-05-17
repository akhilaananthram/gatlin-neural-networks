import argparse
import os
import tables
import numpy as np
import caffe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate encodings from a trained net')
    parser.add_argument("--net", required=True,
      help="Caffemodel to load weights for network")
    parser.add_argument("--solver", default="solver.prototxt",
                        help="path to solver proto file")
    parser.add_argument("--encodings", default="encodings.h5",
                        help="path to save encodings")

    args = parser.parse_args()

    solver = caffe.SGDSolver(args.solver)
    solver.net.copy_from(args.net) 

    n_train_iter = 4500  
    n_val_iter = 1000

    train_blob_dims = solver.net.blobs['probability_dist'].data.shape
    val_blob_dims = solver.test_nets[0].blobs['probability_dist'].data.shape 
    train_dims = (n_train_iter * train_blob_dims[0], train_blob_dims[1])
    val_dims = (n_val_iter * val_blob_dims[0], val_blob_dims[1])

    print 'creating output .h5 file: {}'.format(args.encodings)
    f = tables.open_file(args.encodings, mode='w')
    data_atom = tables.Atom.from_dtype(np.dtype('Float32'))
    print 'initializing output arrays'
    train_array = f.create_array(f.root, 'train', atom=data_atom,
                                 shape=train_dims)
    val_array = f.create_array(f.root, 'val', atom=data_atom,
                               shape=val_dims)

    batch_size = train_blob_dims[0]
    for i in xrange(n_train_iter):
        solver.net.forward()
        train_array[(i*batch_size):(i+1)*batch_size] = solver.net.blobs['probability_dist'].data

    batch_size = val_blob_dims[0]
    for i in xrange(n_val_iter):
        solver.test_nets[0].forward()
        val_array[(i*batch_size):(i+1)*batch_size] = solver.test_nets[0].blobs['probability_dist'].data

    f.close()
    print 'done'
