#!/usr/bin/python3
import argparse
import json
import cv2 as cv
import os
import sys

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

script_path = sys.argv[0] #get_script_path()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help = "Input File")
parser.add_argument("-it", "--input_type", help = "Input File")
args = parser.parse_args()
    
net = cv.dnn_DetectionModel(script_path+'/resources/yolov4.cfg', script_path+'/resources/yolov4.weights')

def imageArray(input,output):
    frame = cv.imread(input)
    height_v, width_v, color_v = frame.shape
    net.setInputSize(704, 704)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB (True)

    with open(script_path+'/resources/coco.names', 'rt') as f:
        names = f.read().rstrip('\n').split('\n')

    classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)

    image_array = {}

    for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
        label = '%.2f' % confidence
        label = '%s: %s' % (names[classId], label) 
        label_name = names[classId]

        labelSize, baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1) 
        left, top, width, height = box 
        
        if label_name in image_array:
            old_value = image_array[ label_name ]
            old_value.append( [str(confidence),str(box)] )
            image_array[ label_name ] = old_value
        else:
            image_array[ label_name ] = [ [str(confidence),str(box)] ]

        top = max(top, labelSize[1]) 
        cv.rectangle(frame, box, color=(0, 255, 0), thickness=3) 
        cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseline), (255, 255, 255), cv.FILLED) 
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imwrite(output, frame)

    return image_array

def getFrameCount(input):
    vidcap = cv.VideoCapture(input)
    fps = vidcap.get(cv.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
    return frame_count


def videoArray(input,sample_size=5):
    vid_file = 'io/1.mp4'
    frame_count = getFrameCount(vid_file)
    vidcap = cv.VideoCapture(vid_file)
    success,frame = vidcap.read()
    frame_interval = int(frame_count/sample_size)
    frame_no = frame_interval
    vid_array = []
    while frame_no <= frame_count:
        vidcap.set( int(vidcap.get(cv.CAP_PROP_POS_FRAMES)) , frame_no)
        # print("Frame No = ",frame_no)
        susccess, frame = vidcap.read()
        cv.imwrite("io/in.jpg" , frame) 
        vid_array.append( imageArray('io/in.jpg','io/img.png') )
        frame_no += frame_interval
    return vid_array
    

if args.input and args.input_type:
    input = args.input
    if args.input_type == "video":
        vidInfo = videoArray(input)
        json_object = json.dumps(vidInfo, indent = 0, separators=(',', ':'))
        json_object = json_object.replace('\n', '')
        print(json_object)
    elif args.input_type == "image":
        imageInfo = imageArray(input,'io/ouput.jpg')
        json_object = json.dumps(imageInfo, indent = 0, separators=(',', ':'))
        json_object = json_object.replace('\n', '')
        print(json_object)
else:
    print("No input received !")
