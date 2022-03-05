import json
import cv2 as cv

net = cv.dnn_DetectionModel('resources/yolov4.cfg', 'resources/yolov4.weights')

frame = cv.imread('io/resturant.jpg')
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

cv.imwrite('io/img.png', frame)

json_object = json.dumps(output_dict, indent = 0, separators=(',', ':'))
json_object = json_object.replace('\n', '')
print(json_object)