from imageai.Detection import ObjectDetection
import os
import time

def identity_object():
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "filename.jpg"),
                                                 output_image_path=os.path.join(execution_path, "output.jpeg"))

    for eachObject in detections:
        print(eachObject["name"], " : ", eachObject["percentage_probability"])


cam = VideoCapture(0)   # 0 -> index of camera
while True:
    s, img = cam.read()
    if s:    # frame captured without any errors
        namedWindow("cam-test",500)
        imshow("cam-test",img)
        waitKey(0)
        destroyWindow("cam-test")
        imwrite("filename.jpg",img) #save image
        identity_object()
        print("Focusing")
        time.sleep(5)
