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

for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
    label = '%.2f' % confidence
    label = '%s: %s' % (names[classId], label) 
    labelSize, baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1) 
    left, top, width, height = box 
    top = max(top, labelSize[1]) 
    cv.rectangle(frame, box, color=(0, 255, 0), thickness=3) 
    cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseline), (255, 255, 255), cv.FILLED) 
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

cv.imwrite('io/img.png', frame)