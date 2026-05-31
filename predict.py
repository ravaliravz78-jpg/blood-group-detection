import cv2
import numpy as np

from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model/blood_group_model.h5")

# Blood group labels
classes = ["A", "B", "AB", "O"]

# Enter fingerprint image path
image_path = input("Enter fingerprint image path: ")

# Read image
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image exists
if img is None:
    print("Image not found!")
    exit()

# Resize image
img = cv2.resize(img, (128, 128))

# Normalize image
img = img / 255.0

# Reshape for CNN
img = img.reshape(1, 128, 128, 1)

# Predict
prediction = model.predict(img)

# Get highest probability index
predicted_class = np.argmax(prediction)

# Final output
print("\nPredicted Blood Group:", classes[predicted_class])