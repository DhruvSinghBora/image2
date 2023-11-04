import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize the camera
camera = cv2.VideoCapture(0)  # Use 0 for the first camera, 1 for the second, and so on

# Check if the camera is opened successfully
if not camera.isOpened():
    print("Error: Could not open the camera")
else:
    # Capture a single frame from the camera
    ret, frame = camera.read()

    if ret:
        # Convert the frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use MediaPipe to detect hand landmarks
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # Draw landmarks on the frame
            for landmarks in results.multi_hand_landmarks:
                for landmark in landmarks.landmark:
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Save the annotated image as "annotated_image.jpg"
        cv2.imwrite("annotated_image.jpg", frame)
        print("Annotated image captured and saved as 'annotated_image.jpg'")
    else:
        print("Error: Could not capture an image")

    # Release the camera
    camera.release()

# Close all OpenCV windows (if any are open)
cv2.destroyAllWindows()
