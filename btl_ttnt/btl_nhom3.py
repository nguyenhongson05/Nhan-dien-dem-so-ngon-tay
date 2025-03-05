import cv2
import mediapipe as mp
from ultralytics import YOLO

# Loading YOLO model
print("🚀 Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("✅ YOLO model loaded successfully!")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)

# Fingertip landmark IDs in MediaPipe
TIP_IDS = [4, 8, 12, 16, 20]

def identify_hand_gesture(hand_landmarks):
    fingers = [0, 0, 0, 0, 0]
    
    if hand_landmarks.landmark[TIP_IDS[0]].x > hand_landmarks.landmark[TIP_IDS[0] - 1].x:
        fingers[0] = 1
    
    for i in range(1, 5):
        if hand_landmarks.landmark[TIP_IDS[i]].y < hand_landmarks.landmark[TIP_IDS[i] - 2].y:
            fingers[i] = 1
    
    if fingers == [1, 1, 0, 0, 1]:  # Ring and middle fingers folded
        return 6
    elif fingers == [1, 0, 0, 0, 1]:  # Index, middle, and ring folded
        return 7
    elif fingers == [1, 1, 0, 1, 1]:  # Middle finger folded
        return 8
    elif fingers == [0, 1, 1, 0, 1] and hand_landmarks.landmark[TIP_IDS[0]].y > hand_landmarks.landmark[TIP_IDS[3]].y:
        return 9  # Thumb and ring touching
    elif fingers == [0, 1, 0, 1, 1] and hand_landmarks.landmark[TIP_IDS[0]].y > hand_landmarks.landmark[TIP_IDS[2]].y:
        return 10  # Thumb and middle touching
    
    return sum(fingers)

def get_finger_bounding_box(hand_landmarks, frame):
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for i in TIP_IDS:
        x, y = int(hand_landmarks.landmark[i].x * frame.shape[1]), int(hand_landmarks.landmark[i].y * frame.shape[0])
        min_x, min_y = min(min_x, x), min(min_y, y)
        max_x, max_y = max(max_x, x), max(max_y, y)
    
    return min_x - 20, min_y - 20, max_x + 20, max_y + 20

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera!")
    exit()
else:
    print("✅ Camera opened successfully!")

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Cannot read frame from camera!")
        break

    frame = cv2.flip(frame, 1)
    results = model(frame)
    hand_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(hand_rgb)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            gesture = identify_hand_gesture(hand_landmarks)
            cv2.putText(frame, f'Hand Gesture: {gesture}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            x1, y1, x2, y2 = get_finger_bounding_box(hand_landmarks, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'No fingers detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("YOLO Hand Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
