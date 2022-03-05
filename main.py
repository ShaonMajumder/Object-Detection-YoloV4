import json
import cv2 as cv

net = cv.dnn_DetectionModel('resources/yolov4.cfg', 'resources/yolov4.weights')

def imageArray(input,output):
    frame = cv.imread(input)
    height_v, width_v, color_v = frame.shape
    net.setInputSize(704, 704)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB (True)


    with open('resources/coco.names', 'rt') as f:
        names = f.read().rstrip('\n').split('\n')

    classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)


    output_dict = {}

    for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
        label = '%.2f' % confidence
        label = '%s: %s' % (names[classId], label) 
        label_name = names[classId]


        labelSize, baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1) 
        left, top, width, height = box 
        
        if label_name in output_dict:
            old_value = output_dict[ label_name ]
            old_value.append( [str(confidence),str(box)] )
            output_dict[ label_name ] = old_value
        else:
            output_dict[ label_name ] = [ [str(confidence),str(box)] ]

        top = max(top, labelSize[1]) 
        cv.rectangle(frame, box, color=(0, 255, 0), thickness=3) 
        cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseline), (255, 255, 255), cv.FILLED) 
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imwrite(output, frame)

    return output_dict

def getFrameCount(input):
    vidcap = cv.VideoCapture(input)
    fps = vidcap.get(cv.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
    return frame_count


def videoArray(input):
    vid_file = 'io/1.mp4'
    frame_count = getFrameCount(vid_file)
    vidcap = cv.VideoCapture(vid_file)
    success,frame = vidcap.read()
    frame_interval = int(frame_count/10)
    frame_no = frame_interval
    vid_array = []
    while frame_no <= frame_count:
        # vidcap.set(2,frame_no)
        vidcap.set( int(vidcap.get(cv.CAP_PROP_POS_FRAMES)) , frame_no)
        print("Frame No = ",frame_no)
        susccess, frame = vidcap.read()
        cv.imwrite("io/in.jpg" , frame) 
        vid_array.append( imageArray('io/in.jpg','io/img.png') )
        frame_no += frame_interval
    return vid_array
    

vidInfo = videoArray('io/1.mp4')
json_object = json.dumps(vidInfo, indent = 0, separators=(',', ':'))
json_object = json_object.replace('\n', '')
print(json_object)