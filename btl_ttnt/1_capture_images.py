import cv2
import os

# Tạo thư mục lưu dữ liệu nếu chưa có
save_dir = "dataset/images/"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Hand Images", frame)

    # Nhấn 's' để lưu ảnh
    key = cv2.waitKey(1)
    if key == ord('s'):
        filename = os.path.join(save_dir, f"hand_{count}.jpg")
        cv2.imwrite(filename, frame)
        print(f"✅ Saved: {filename}")
        count += 1

    # Nhấn 'q' để thoát
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
