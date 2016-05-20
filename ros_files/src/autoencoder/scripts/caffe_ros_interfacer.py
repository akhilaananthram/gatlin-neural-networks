#!/usr/bin/env python
import argparse
import caffe
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3

def process_image(model, weights, blob_name, in_topic, out_topic):
    # Uncomment when training with a GPU
    caffe.set_mode_cpu()
    #caffe.set_device(0)
    #caffe.set_mode_gpu()

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
        blobs = caffe_blobs[blob_name]

        # TODO: If you want the net to publish something else, change this loop
        # and all references to Vector3 to match the message you want
        xs = blobs[::2]
        ys = blobs[1::2]
        for x, y in zip(xs, ys):
            # To publish image coordinates use the following condition
            #if x >= 0 and y >= 0:
            out_msg = Vector3()
            out_msg.x = x
            out_msg.y = y
            pub.publish(out_msg)

    # publish to topic
    pub = rospy.Publisher(out_topic, Vector3, queue_size=10)

    rospy.init_node('caffe_interfacer', anonymous=True)

    rospy.Subscriber(in_topic, Image, callback)

    # spin() simply keeps python from exiting until this node is stopped
    # Does not affect callback function as that has its own thread
    rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates blobs for image")
    parser.add_argument("--model", required=True, help="path to deploy.prototxt")
    parser.add_argument("--weights", required=True, help="path to *.caffemodel")
    parser.add_argument("--blob-name", default="blob_points",
                        help="Name of caffe blob to publish")
    parser.add_argument("--in-topic", default="/camera/rgb/image_rect_color_throttled",
                        help="Name of topic to listen to")
    parser.add_argument("--out-topic", default="/gatlin/objectscreencoords",
                        help="Name of topic to publish to")
    args = parser.parse_args()

    process_image(args.model, args.weights, args.blob_name, args.in_topic, args.out_topic)
