from ultralytics import YOLO
import os

model = YOLO('yolov8n.pt')

data_yaml = 'data.yaml'

model.train(data=data_yaml,
            epochs=100,
            imgsz=416,
            batch=4,
            name='yolo_training',
            project='runs/train')