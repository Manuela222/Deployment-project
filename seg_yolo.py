from ultralytics import YOLO 


model = YOLO('yolov8n-seg.pt')
model.train(data = "C:/Users/manue/OneDrive/Documents/4e année/Deployment of an AI model/group project/Chip_Segmentation.v1i.yolov8/data.yaml",
            epochs = 20,
            batch = 8
            )
