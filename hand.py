import cv2
import mediapipe as mp
import math

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            lm_list = []
            h, w, c = frame.shape
            for id, lm in enumerate(handLms.landmark):
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            if lm_list:
                # Thumb tip = 4, Index tip = 8
                x1, y1 = lm_list[4]
                x2, y2 = lm_list[8]

                # Draw circles on thumb and index
                cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 10, (255, 0, 255), cv2.FILLED)

                # Calculate distance
                distance = math.hypot(x2 - x1, y2 - y1)

                # Threshold for "joined fingers"
                if distance < 40:
                    color = (255, 0, 0)  # Blue
                else:
                    color = (0, 0, 255)  # Red

                # Display color circle on corner
                cv2.circle(frame, (50, 50), 30, color, cv2.FILLED)

    cv2.imshow("Finger Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
