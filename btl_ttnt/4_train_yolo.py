from ultralytics import YOLO

model = YOLO("yolov8n.pt")  
model.train(data="D:/V/btl_ttnt/dataset.yaml", epochs=50, imgsz=640)  # Dùng đường dẫn đầy đủ
