from ultralytics import YOLO 
import matplotlib.pyplot as plt
import cv2

model = YOLO('runs/segment/train/weights/best.pt')
def validation(image_path):
    results = model.predict(source = image_path, conf=0.25, save=True)
    labeled_image = cv2.imread(f"C:/Users/manue/OneDrive/Documents/4e année/Deployment of an AI model/group project/runs/segment/predict/{image_path.split('/')[-1]}")
    labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10,10))
    plt.imshow(labeled_image)
    plt.title("Prediction")
    plt.axis('off')
    plt.show()

validation('C:/Users/manue/OneDrive/Documents/4e année/Deployment of an AI model/group project/Chip_Segmentation.v1i.yolov8/valid/images/06_jpg.rf.2adaf3176a3250246148d7cdc34c7709.jpg')