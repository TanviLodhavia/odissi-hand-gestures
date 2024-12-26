import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from config import *

# Load the trained model
# Load the trained model
model_path = 'model_checkpoint.keras'
model = tf.keras.models.load_model(model_path)

class_names = ['Alapadma','Chandrakala','Chatura','Hamsasya','Mushti','Padmakosha','Pataka','Suchi','Tamarachuda','Trisula'
                ]

# initialize the mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hands in the frame
    results = hands.process(rgb_frame)

    try:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box of the hand
                x_min, y_min, x_max, y_max = 1.0, 1.0, 0.0, 0.0
                for landmark in hand_landmarks.landmark:
                    x_min = min(x_min, landmark.x)
                    x_max = max(x_max, landmark.x)
                    y_min = min(y_min, landmark.y)
                    y_max = max(y_max, landmark.y)

                # Convert normalized coordinates to pixel coordinates
                height, width, _ = frame.shape
                x_min = int(x_min * width) - 30
                x_max = int(x_max * width) + 20
                y_min = int(y_min * height) - 30
                y_max = int(y_max * height) + 20  

                # Crop and preprocess the hand region
                hand_image = frame[y_min:y_max, x_min:x_max]
                resized_hand = cv2.resize(hand_image, (150, 150))
                resized_hand = image.img_to_array(resized_hand)
                preprocessed_hand = np.expand_dims(resized_hand, axis=0)
                preprocessed_hand = preprocess_input(preprocessed_hand)
                
                # Make predictions only if hand is detected
                predictions = model.predict(preprocessed_hand)
                print(predictions)
                print(np.argmax(predictions))
                predicted_mudra = class_names[np.argmax(predictions)]
                
                # Display the predicted mudra on the frame
                cv2.putText(frame, predicted_mudra, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    except Exception as e:
        pass

    # Display the frame with predictions
    cv2.imshow("Gesture Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()