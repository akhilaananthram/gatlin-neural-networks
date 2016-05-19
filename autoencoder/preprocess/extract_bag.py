import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from rosbag.bag import Bag
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save data from bag file")
    parser.add_argument("--bag", required=True, help="Path to bag file")
    parser.add_argument("--output", default="bag_output", help="directory for data")
    parser.add_argument("--blob-topic", dest="blob_topic",
                        default="/gatlin/objectscreencoords",
                        help="topic for blobs")
    parser.add_argument("--image-topic", dest="image_topic",
                        default="/camera/rgb/image_rect_color_throttled",
                        help="topic for images")
    parser.add_argument("--factor", default=10**7, help="Factor for decimal digits")
    args = parser.parse_args()

    bag = Bag(args.bag)
    messages = bag.read_messages()

    # Convert bag data to correct format
    groups = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
    blobs_to_group_later = []
    for topic, msg, time in messages:
        if topic == args.blob_topic:
            data = [msg.x, msg.y]
            blobs_to_group_later.append((data, time))
        else:
            data = np.fromstring(msg.data, np.uint8)
            data = np.reshape(data, (msg.height, msg.width, 3))
            groups[time.secs][int(time.nsecs / args.factor)][args.image_topic].append(data)

    # Match blobs with images
    for data, time in blobs_to_group_later:
        actual_time = int(time.nsecs / args.factor)
        possible_times = groups[time.secs].keys()
        possible_times = sorted(possible_times, key=lambda x: abs(x - actual_time))
        actual_time = possible_times[0]
        groups[time.secs][actual_time][args.blob_topic] += data

    # Mark new blobs
    for sec, times in groups.iteritems():
        for nsec, data in times.iteritems():
            img = data[args.image_topic][0]
            height, _, _ = img.shape
            blobs = data[args.blob_topic]
            imgplt = plt.imshow(img)
            plt.scatter(blobs[::2], blobs[1::2])
            fig = plt.figure(1) # Get current figure
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
            plt.axis("off")
            def onclick(event, blobs=blobs, height=height):
                print event.x, height - event.y
                blobs += [event.x, height - event.y]
                plt.scatter(blobs[::2], blobs[1::2])
                plt.draw()
            fig.canvas.mpl_connect("button_press_event", onclick)
            plt.show()

    # Pad blobs to max length
    max_blob_count = 0
    for sec, times in groups.iteritems():
        for nsec, data in times.iteritems():
            max_blob_count = max(max_blob_count, len(data[args.blob_topic]))

    for sec, times in groups.iteritems():
        for nsec, data in times.iteritems():
            num_blobs = len(data[args.blob_topic])
            data[args.blob_topic] += [-1] * (max_blob_count - num_blobs)

    # Save files
    if args.output is not None:
        if not os.path.isdir(args.output):
            os.mkdir(args.output)

        with open(os.path.join(args.output, "blobs.txt"), "w") as f:
            for sec, times in groups.iteritems():
                for nsec, data in times.iteritems():
                    img = data[args.image_topic][0]
                    blobs = data[args.blob_topic]
                    img_name = os.path.join(args.output, "{}{}.jpg".format(sec, nsec))
                    cv2.imwrite(img_name, img)
                    f.write("{}{},{}\n".format(sec, nsec, " ".join([str(round(b, 3)) for b in blobs])))
