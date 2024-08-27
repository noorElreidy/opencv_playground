import cv2
import mediapipe as mp

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize hands detector
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

def count_fingers(hand_landmarks):
    # Count fingers based on landmarks positions
    finger_tips = [4, 8, 12, 16, 20]  # Indexes of the fingertip landmarks
    count = 0
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    return count

# Start capturing video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            num_fingers = count_fingers(hand_landmarks)
            cv2.putText(frame, f'Fingers: {num_fingers}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Finger Counting', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
