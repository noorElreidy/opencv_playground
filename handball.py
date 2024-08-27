import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

ball_radius = 20
ball_color = (0, 0, 255)  # Red
ball_position = np.array([320, 240], dtype=float)
ball_velocity = np.array([0, 0], dtype=float)
gravity = 0.5

cap = cv2.VideoCapture(0)

def get_hand_center(landmarks):
    if landmarks:
        x_coords = [landmark.x for landmark in landmarks.landmark]
        y_coords = [landmark.y for landmark in landmarks.landmark]
        center_x = int(np.mean(x_coords) * frame_width)
        center_y = int(np.mean(y_coords) * frame_height)
        return np.array([center_x, center_y])
    return None

def is_thumbs_up(hand_landmarks):
    if not hand_landmarks:
        return False
    
    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    
    finger_tips = [landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                   landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                   landmarks[mp_hands.HandLandmark.RING_FINGER_TIP],
                   landmarks[mp_hands.HandLandmark.PINKY_TIP]]
    finger_bases = [landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                    landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
                    landmarks[mp_hands.HandLandmark.RING_FINGER_MCP],
                    landmarks[mp_hands.HandLandmark.PINKY_MCP]]
    
    thumb_extended = thumb_tip.y < thumb_ip.y
    fingers_curl = all(tip.y > base.y for tip, base in zip(finger_tips, finger_bases))
    
    return thumb_extended and fingers_curl

def is_grabbing(hand_landmarks):
    if not hand_landmarks:
        return False

    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    
    grab_threshold = 0.05
    
    distance_thumb_index = abs(thumb_tip.x - index_tip.x) + abs(thumb_tip.y - index_tip.y)
    distance_thumb_middle = abs(thumb_tip.x - middle_tip.x) + abs(thumb_tip.y - middle_tip.y)
    
    return distance_thumb_index < grab_threshold and distance_thumb_middle < grab_threshold

game_started = False
ball_held = False
thumbs_up_count = 0

window_name = 'Hold and Drop the Ball'
cv2.namedWindow(window_name)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_height, frame_width, _ = frame.shape

    # Flip frame so its not confusing 
    frame = cv2.flip(frame, 1)

    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_center = None
    thumbs_up_detected = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_center = get_hand_center(hand_landmarks)

            if is_thumbs_up(hand_landmarks):
                thumbs_up_detected = True

    if thumbs_up_detected:
        thumbs_up_count += 1
        if thumbs_up_count == 2:
            break

    if game_started:
        if is_grabbing(hand_landmarks):
            ball_held = True
  
            ball_velocity = np.array([0, 0])
            ball_position = hand_center  
        else:
            ball_held = False

        if not ball_held:
            ball_velocity[1] += gravity
            ball_position += ball_velocity

            # Bounce the ball off the edges of the frame
            if ball_position[0] < ball_radius or ball_position[0] > frame_width - ball_radius:
                ball_velocity[0] = -ball_velocity[0]
            if ball_position[1] < ball_radius or ball_position[1] > frame_height - ball_radius:
                ball_velocity[1] = -ball_velocity[1]

            cv2.circle(frame, (int(ball_position[0]), int(ball_position[1])), ball_radius, ball_color, -1)
    else:
        if thumbs_up_detected:
            game_started = True

    cv2.imshow(window_name, frame)


    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
