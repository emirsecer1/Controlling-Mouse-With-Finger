import cv2
import mediapipe as mp
import pyautogui
import math

# MediaPipe
pyautogui.FAILSAFE = False
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load TensorFlow model
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Capture video
cap = cv2.VideoCapture(0)

# Finger distance thresholds (Adjust!)
click_threshold = 0.03
drag_threshold = 0.02
scroll_threshold = 0.03

# Fingertip position variables
prev_finger_x, prev_finger_y = None, None
sensitivity = 2.0

# Smoothing
prev_x, prev_y = None, None
smoothing_factor = 3

# Dragging state
dragging = False

# Scrolling state
scrolling = False

# Time interval to determine key hold and release
PRESS_TIME_THRESHOLD = 0.2  # Seconds
last_press_time = 0

# Button positions
left_button_pos = (70, 240)
right_button_pos = (470, 240)
up_button_pos = (270, 140)
down_button_pos = (270, 340)
button_radius = 20


def is_inside_circle(point, center, radius):
  return math.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2) <= radius

def draw_button(image, pos, text):
    cv2.circle(image, pos, button_radius, (255, 0, 0), -1)
    cv2.putText(image, text, (pos[0] - 10, pos[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Video capture failed.")
        break

    # Convert image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_x, thumb_y = hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y
            index_x, index_y = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
            middle_x, middle_y = hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y
            ring_x, ring_y = hand_landmarks.landmark[16].x, hand_landmarks.landmark[16].y

            # Distances
            thumb_index_dist = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5
            thumb_middle_dist = ((thumb_x - middle_x) ** 2 + (thumb_y - middle_y) ** 2) ** 0.5
            thumb_ring_dist = ((thumb_x - ring_x) ** 2 + (thumb_y - ring_y) ** 2) ** 0.5

            # Click Management
            if thumb_middle_dist < click_threshold:
                    pyautogui.click()

            if thumb_ring_dist < click_threshold:
                pyautogui.rightClick()

            # Drag Management
            is_drag_mode = thumb_index_dist < drag_threshold and thumb_middle_dist < drag_threshold and thumb_ring_dist < drag_threshold

            if is_drag_mode:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            # Scroll Mode Detection
            dx, dy = index_x - thumb_x, index_y - thumb_y
            thumb_angle = math.atan2(dy, dx)
            is_scroll_mode = abs(thumb_angle) > math.pi / 3

            # Scrolling
            if is_scroll_mode and thumb_index_dist < scroll_threshold:
                if not scrolling:
                    scrolling = True
                scroll_speed = 200
                if thumb_angle >= 0:
                    pyautogui.scroll(scroll_speed)
                else:
                    pyautogui.scroll(-scroll_speed)
            else:
                scrolling = False

            # Motion Calculation
            if prev_finger_x is None and prev_finger_y is None:
                prev_finger_x, prev_finger_y = index_x, index_y

            # Motion Calculation
            displacement_x = (index_x - prev_finger_x) * sensitivity
            displacement_y = (index_y - prev_finger_y) * sensitivity

            # Updating the Mouse Position
            if results.multi_hand_landmarks and not dragging:
                pyautogui.moveRel(displacement_x, displacement_y)

            # Update current position as new reference
            prev_finger_x, prev_finger_y = index_x, index_y

            # Draw landmarks (optional)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw buttons
    draw_button(frame, left_button_pos, "Left")
    draw_button(frame, right_button_pos, "Right")
    draw_button(frame, up_button_pos, "Up")
    draw_button(frame, down_button_pos, "Down")
  
    # Check for button clicks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_x, index_y = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
            if is_inside_circle((index_x * frame.shape[1], index_y * frame.shape[0]), left_button_pos, button_radius):
                pyautogui.moveRel(-10, 0)
            elif is_inside_circle((index_x * frame.shape[1], index_y * frame.shape[0]), right_button_pos, button_radius):
                pyautogui.moveRel(10, 0)
            elif is_inside_circle((index_x * frame.shape[1], index_y * frame.shape[0]), up_button_pos, button_radius):
                pyautogui.moveRel(0, -10)
            elif is_inside_circle((index_x * frame.shape[1], index_y * frame.shape[0]), down_button_pos, button_radius):
                pyautogui.moveRel(0, 10)
            

    # Display frame
    cv2.imshow('Hand Gesture Control', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
