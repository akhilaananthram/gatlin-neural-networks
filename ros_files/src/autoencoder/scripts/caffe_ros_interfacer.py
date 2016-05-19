#!/usr/bin/env python
import argparse
import caffe
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3

def process_image(model, weights):
    # publish to topic
    pub = rospy.Publisher('/gatlin/objectscreencoords', Vector3, queue_size=10)
    rospy.init_node('object_screen_coordinate_maker', anonymous=True)

    # TODO: use GPU if available
    caffe.set_mode_cpu()

    net = caffe.Net(model, weights, caffe.TEST)
    # only want to run one image at a time, so reshape
    _, C, H, W = net.blobs["original"].data.shape
    shape = (1, C, H, W)
    net.blobs['original'].reshape(*(shape))
    net.reshape()

    def callback(in_msg, pub=pub, net=net):
        img = np.fromstring(in_msg.data, np.uint8)
        img = np.reshape(img, (in_msg.height, in_msg.width, 3))
        # (H, W, C) -> (C, H, W)
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 0, 2)

        # run through net
        caffe_blobs = net.forward(original=img)
        blobs = caffe_blobs["blob_points"]

        # Only publish valid blobs
        xs = blobs[::2]
        ys = blobs[1::2]
        for x, y in zip(xs, ys):
            if x >= 0 and y >= 0:
                out_msg = Vector3()
            out_msg.x = x
            out_msg.y = y
            pub.publish(out_msg)

    rospy.init_node('image_handler', anonymous=True)

    rospy.Subscriber('/camera/rgb/image_rect_color_throttled', Image, callback)

    # spin() simply keeps python from exiting until this node is stopped
    # Does not affect callback function as that has its own thread
    rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates blobs for image")
    parser.add_argument("--model", required=True, help="path to deploy.prototxt")
    parser.add_argument("--weights", required=True, help="path to *.caffemodel")
    args = parser.parse_args()

    process_image(args.model, args.weights)
