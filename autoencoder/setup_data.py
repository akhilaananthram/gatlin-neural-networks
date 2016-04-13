"""
Modified http://deepdish.io/2015/04/28/creating-lmdb-in-python/
"""
import argparse
import caffe
import lmdb
import numpy as np
import os
import random
import sys

from PIL import Image # Pillow

def prepare_database(directory, images):
    # If directory does not exist, create it
    if not os.path.isdir(directory):
        os.mkdir(directory)

    n_bytes = sum([os.path.getsize(os.path.join(args.data, img)) for img in images])
    print "Setting up {} images".format(len(images))

    # We need to prepare the database for the size. We'll set it 30 times
    # greater than what we theoretically need. There is little drawback to
    # setting this too big. If you still run into problem after raising
    # this, you might want to try saving fewer entries in a single
    # transaction.
    return lmdb.open(directory, map_size=n_bytes*30)

def fill_database(env, images):
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i, img_path in enumerate(images):
            img = np.array(Image.open(os.path.join(args.data, img_path)))

            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = img.shape[2]
            datum.height = img.shape[1]
            datum.width = img.shape[0]
            datum.data = img.tobytes()  # or .tostring() if numpy < 1.9
            datum.label = i
            str_id = '{:08}'.format(i)

            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

def save_file(path, images, dest):
    with open(dest, 'w') as f:
        for filename in images:
            f.write("{} 0\n".format(os.path.join(path, filename)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="creates database or caffe formatted text files from images")
    parser.add_argument("--data", default="data", help="Folder containing images")
    parser.add_argument("--db", default="img_db", help="Folder to contain DB")
    parser.add_argument("--out-train", default="train.txt", help="File containing paths to images")
    parser.add_argument("--out-val", default="val.txt", help="File containing paths to images")
    parser.add_argument("--randomize", action="store_true", help="Randomize the data")
    parser.add_argument("--val", default=0.2, help="Percentage for validation")
    parser.add_argument("--format", choices=["db", "txt"], default="txt", help="Format to save data in")
    args = parser.parse_args()

    # Get image paths and size
    images = np.array([f for f in os.listdir(args.data) if not f.startswith(".")])

    # Divide into validation and training
    num_validation = int(len(images) * args.val)
    val_indices = np.zeros(images.shape, dtype=bool)
    val_indices[random.sample(xrange(len(images)), num_validation)] = True
    val_images = images[val_indices]
    train_images = images[~val_indices]

    if args.randomize:
        np.random.shuffle(val_images)
        np.random.shuffle(train_images)

    if args.format == "db":
        env_train = prepare_database(os.path.join(args.db, "train"), train_images)
        env_val = prepare_database(os.path.join(args.db, "val"), val_images)

        fill_database(env_train, train_images)
        fill_database(env_val, val_images)
    else:
        save_file(args.data, train_images, args.out_train)
        save_file(args.data, val_images, args.out_val)
