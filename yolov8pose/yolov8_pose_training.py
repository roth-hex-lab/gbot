from ultralytics import YOLO

# Load a model
model = YOLO('YOLOv8x-pose-p6')  # load a pretrained model (recommended for training)


# Train the model
results = model.train(data='yolov8/config/CornerClamp.yaml', epochs=300, imgsz=1280)