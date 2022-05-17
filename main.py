import cv2
from random import randint

file_name="coc names.py"
with open(file_name,'rt')as fpt:
    class_name=fpt.read().rstrip('\n').split('\n')



class_color = []
for i in range(len(class_name)):
    class_color.append((randint(0,255),randint(0,255),randint(0,255)))

config_file="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model="frozen_inference_graph.pb"
net=cv2.dnn_DetectionModel(frozen_model, config_file)



net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)
cap=cv2.VideoCapture("Road_traffic_video2.mp4")
while True:
    success, img = cap.read()

    ClassIds, confs,bbox=net.detect(img,confThreshold=0.5)
    if len((ClassIds) != 0):
        for ClassId, confidence, box in zip(ClassIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=class_color[ClassId-1], thickness=1)
            cv2.putText(img, class_name[ClassId-1].upper(), (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, class_color[ClassId-1], 2)
    cv2.imshow("output", img)

    if cv2.waitKey(1)&0xff == ('q'):
        break


