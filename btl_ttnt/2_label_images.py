import cv2
import os
from ultralytics import YOLO
import mediapipe
# Load YOLO pre-trained model
model = YOLO("yolov8n.pt")

# Thư mục dữ liệu
img_dir = "dataset/images/"
label_dir = "dataset/labels/"
os.makedirs(label_dir, exist_ok=True)

for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)

    results = model(img)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            w, h = img.shape[1], img.shape[0]

            # Chuyển sang định dạng YOLO (tọa độ chuẩn hóa)
            x_center = (x1 + x2) / (2 * w)
            y_center = (y1 + y2) / (2 * h)
            width = (x2 - x1) / w
            height = (y2 - y1) / h

            # Ghi nhãn vào file .txt
            label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))
            with open(label_path, "w") as f:
                f.write(f"0 {x_center} {y_center} {width} {height}\n")

print("✅ Hoàn tất gán nhãn tự động!")
