import time

from ultralytics import YOLO

if __name__ == '__main__':
   model = YOLO("yolov8x-pose-p6.pt")  # load an official model
   model = YOLO("pathtoyourmodel/best.pt", task="pose") # Continue Training, set resume in train method to True

   # https://docs.ultralytics.com/modes/predict/
   results = model("pathtoyourimage", imgsz=(1280, 1280), stream=False, conf=0.3, show=True)  # predict on an image
   time.sleep(20)
