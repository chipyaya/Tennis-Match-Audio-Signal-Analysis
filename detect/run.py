import argparse
import glob
import cv2
import argparse
import numpy as np

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--directory', required=True, help = 'path to image directory')
args = ap.parse_args()
images = glob.glob(args.directory+'/**/*.png', recursive=True)
classes = None
with open("classes.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
for k, img in enumerate(images):
    image = cv2.imread(img)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    output_name = img[:-4] + "_human.txt"
    print(output_name + " {}/{} ({:.2f}%)".format(k+1, len(images), 100*(k+1)/len(images)))
    with open(output_name, 'w') as f:
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            f.write("{} {} {} {}\n".format(x, y, w, h))
