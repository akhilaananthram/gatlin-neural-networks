"""
Modified http://deepdish.io/2015/04/28/creating-lmdb-in-python/
"""
import argparse
import caffe
import lmdb
import os

from PIL import Image # Pillow

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="creates database from images")
    parser.add_argument("--data", default="data", help="Folder containing images")
    parser.add_argument("--db", default="img_db", help="Folder to contain DB")
    args = parser.parse_args()

    # Get image paths and size
    images = [name for name in os.listdir(args.data) if os.path.isfile(name)]
    n_bytes = sum([os.path.getsize(img) for img in images])

    # We need to prepare the database for the size. We'll set it 10 times
    # greater than what we theoretically need. There is little drawback to
    # setting this too big. If you still run into problem after raising
    # this, you might want to try saving fewer entries in a single
    # transaction.
    env = lmdb.open(args.db, map_size=n_bytes*10)

    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i, img_path in enumerate(images):
            img = np.array(Image.open(img_path))

            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = img.shape[2]
            datum.height = img.shape[1]
            datum.width = img.shape[0]
            datum.data = img.tobytes()  # or .tostring() if numpy < 1.9
            datum.label = i
            str_id = '{:08}'.format(i)

            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
