import cv2
import mediapipe as mp
import os
from ultralytics import YOLO

# Load mô hình YOLO đã huấn luyện
print("🚀 Đang tải mô hình YOLO...")
model = YOLO("runs/detect/train4/weights/best.pt")
print("✅ Mô hình YOLO đã tải xong!")

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)

# Danh sách ID của các đầu ngón tay trong MediaPipe
TIP_IDS = [4, 8, 12, 16, 20]

def identify_hand_gesture(hand_landmarks):
    fingers = [0, 0, 0, 0, 0]
    
    if hand_landmarks.landmark[TIP_IDS[0]].x > hand_landmarks.landmark[TIP_IDS[0] - 1].x:
        fingers[0] = 1
    
    for i in range(1, 5):
        if hand_landmarks.landmark[TIP_IDS[i]].y < hand_landmarks.landmark[TIP_IDS[i] - 2].y:
            fingers[i] = 1
    
    if fingers == [1, 1, 0, 0, 1]:  # Ngón áp út và giữa gập xuống
        return 6
    elif fingers == [1, 0, 0, 0, 1]:  # Chỉ giơ ngón cái và út
        return 7
    elif fingers == [1, 1, 0, 1, 1]:  # Gập ngón giữa
        return 8
    elif fingers == [0, 1, 1, 0, 1] and hand_landmarks.landmark[TIP_IDS[0]].y > hand_landmarks.landmark[TIP_IDS[3]].y:
        return 9  # Ngón cái chạm ngón áp út
    elif fingers == [0, 1, 0, 1, 1] and hand_landmarks.landmark[TIP_IDS[0]].y > hand_landmarks.landmark[TIP_IDS[2]].y:
        return 10  # Ngón cái chạm ngón giữa
    
    return sum(fingers)

def get_finger_bounding_box(hand_landmarks, frame):
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for i in TIP_IDS:
        x, y = int(hand_landmarks.landmark[i].x * frame.shape[1]), int(hand_landmarks.landmark[i].y * frame.shape[0])
        min_x, min_y = min(min_x, x), min(min_y, y)
        max_x, max_y = max(max_x, x), max(max_y, y)
    
    return min_x - 20, min_y - 20, max_x + 20, max_y + 20

# Tạo thư mục lưu ảnh nếu chưa có
save_dir = "captured_images/"
os.makedirs(save_dir, exist_ok=True)

count = 0  # Biến đếm ảnh

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể mở camera!")
    exit()
else:
    print("✅ Camera đã mở thành công!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Không thể đọc frame từ camera!")
        break
    
    frame = cv2.flip(frame, 1)
    results = model(frame)
    hand_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(hand_rgb)
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            gesture = identify_hand_gesture(hand_landmarks)
            cv2.putText(frame, f'Finger number: {gesture}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            x1, y1, x2, y2 = get_finger_bounding_box(hand_landmarks, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'No finger', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Nhận diện tay và cử chỉ", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):  # Lưu ảnh khi ấn 's'
        filename = os.path.join(save_dir, f"captured_{count}.jpg")
        cv2.imwrite(filename, frame)
        print(f"✅ Ảnh đã lưu: {filename}")
        count += 1

cap.release()
cv2.destroyAllWindows()
