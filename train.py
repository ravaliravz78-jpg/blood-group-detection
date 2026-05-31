import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("TRAINING STARTED")

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Dataset path
dataset_path = "dataset"

# Blood group classes
classes = ["A", "B", "AB", "O"]

# Data storage
data = []
labels = []

# Load images
for label, folder in enumerate(classes):

    folder_path = os.path.join(dataset_path, folder)

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (128, 128))

        # Normalize
        img = img / 255.0

        data.append(img)
        labels.append(label)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Reshape for CNN
data = data.reshape(-1, 128, 128, 1)

# Convert labels
labels = to_categorical(labels, num_classes=4)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42
)

# Build CNN Model
model = Sequential()

# Layer 1
model.add(Conv2D(32, (3,3),
                 activation='relu',
                 input_shape=(128,128,1)))

model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 2
model.add(Conv2D(64, (3,3),
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten
model.add(Flatten())

# Dense Layer
model.add(Dense(128, activation='relu'))

# Output Layer
model.add(Dense(4, activation='softmax'))

# Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/blood_group_model.h5")

print("Model trained and saved successfully!")

# Plot Accuracy Graph
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()