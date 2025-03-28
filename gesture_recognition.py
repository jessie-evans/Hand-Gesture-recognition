import cv2
import mediapipe as mp
import time
import screen_brightness_control as sbc
from pynput.keyboard import Key, Controller

# Initialize Mediapipe Hand Module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize keyboard controller for tab switching and volume control
keyboard = Controller()

# Camera Capture
cap = cv2.VideoCapture(0)

# Gesture Cooldown (Prevents spam execution)
gesture_cooldown = 2  # seconds
last_gesture_time = time.time()

# Variables for swipe detection
swipe_threshold = 0.2  # Minimum x-axis movement for swipe
swipe_start_x = None
swipe_start_time = None

# Function to change screen brightness
def change_brightness(increase=True):
    """Change screen brightness."""
    current_brightness = sbc.get_brightness()[0]
    new_brightness = min(100, current_brightness + 10) if increase else max(10, current_brightness - 10)
    sbc.set_brightness(new_brightness)
    print(f"🔆 Brightness set to: {new_brightness}%")
    return new_brightness  # Return brightness value to display on-screen

# Function to switch tabs
def switch_tabs():
    """Simulates Alt + Tab to switch applications."""
    keyboard.press(Key.alt)
    keyboard.press(Key.tab)
    keyboard.release(Key.tab)
    keyboard.release(Key.alt)
    print("↔️ Switched tabs!")

# Function to change volume
def change_volume(increase=True):
    """Change volume (Windows only)."""
    keyboard.press(Key.media_volume_up if increase else Key.media_volume_down)
    keyboard.release(Key.media_volume_up if increase else Key.media_volume_down)
    print(f"🔊 Volume {'Increased' if increase else 'Decreased'}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert color
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Initialize gesture
    detected_gesture = "No Gesture Detected"
    brightness_value = sbc.get_brightness()[0]  # Get current brightness

    # Detect hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark positions
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            # Gesture Logic
            fingers_open = all([
                index_tip.y < wrist.y - 0.1, 
                middle_tip.y < wrist.y - 0.1,
                ring_tip.y < wrist.y - 0.1,
                pinky_tip.y < wrist.y - 0.1
            ])
            fist_closed = all([
                index_tip.y > wrist.y + 0.1, 
                middle_tip.y > wrist.y + 0.1,
                ring_tip.y > wrist.y + 0.1,
                pinky_tip.y > wrist.y + 0.1
            ])

            # Brightness control
            if fingers_open:  
                detected_gesture = "Hand Open (Increase Brightness)"
            elif fist_closed:  
                detected_gesture = "Fist (Decrease Brightness)"

            # Volume control
            elif thumb_tip.y < thumb_ip.y:  
                detected_gesture = "Thumbs Up (Increase Volume)"
            elif thumb_tip.y > thumb_ip.y:  
                detected_gesture = "Thumbs Down (Decrease Volume)"

            # **Fixed Swipe Detection (Switch Tabs)**
            current_time = time.time()
            if swipe_start_x is None:  
                swipe_start_x = index_tip.x
                swipe_start_time = current_time
            else:
                swipe_distance = index_tip.x - swipe_start_x
                if current_time - swipe_start_time < 0.5:  # Swipe within 0.5s
                    if swipe_distance > swipe_threshold:  # Swipe Right
                        detected_gesture = "Swipe Right (Switch Tabs)"
                        switch_tabs()
                        swipe_start_x = None  # Reset after detection
                    elif swipe_distance < -swipe_threshold:  # Swipe Left
                        detected_gesture = "Swipe Left (Switch Tabs)"
                        switch_tabs()
                        swipe_start_x = None  # Reset after detection
                else:
                    swipe_start_x = None  # Reset if too much time passed

            # Execute action with cooldown
            if time.time() - last_gesture_time > gesture_cooldown:
                if "Increase Brightness" in detected_gesture:
                    brightness_value = change_brightness(True)
                elif "Decrease Brightness" in detected_gesture:
                    brightness_value = change_brightness(False)
                elif "Increase Volume" in detected_gesture:
                    change_volume(True)
                elif "Decrease Volume" in detected_gesture:
                    change_volume(False)

                last_gesture_time = time.time()  # Reset cooldown

    # Display Gesture and Brightness %
    cv2.putText(frame, detected_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Brightness: {brightness_value}%", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
