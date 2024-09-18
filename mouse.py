import cv2
import mediapipe as mp
import pyautogui

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe hands detector
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Get the screen size
screen_width, screen_height = pyautogui.size()

# Variable to store the y-coordinate of the index finger
index_y = 0

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a more natural interaction
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Convert the frame from BGR to RGB color space
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB frame to detect hands
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            # Draw hand landmarks on the frame
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # Index finger tip
                    cv2.circle(frame, (x, y), 15, (0, 255, 0), -1)
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y
                    # Move the mouse cursor to the calculated position
                    pyautogui.moveTo(index_x, index_y)

                if id == 4:  # Thumb tip
                    cv2.circle(frame, (x, y), 15, (0, 255, 0), -1)
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y
                    print('Distance:', abs(index_y - thumb_y))

                    # Perform a mouse click if the index finger and thumb are close enough
                    if abs(index_y - thumb_y) < 20:
                        pyautogui.click()
                        pyautogui.sleep(1)

    # Display the frame
    cv2.imshow('Virtual Mouse', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
