import tensorflow as tf
from keras.src.layers import Dropout
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score
import cv2
import numpy as np
import os

# Function to load and preprocess images
def load_and_preprocess_data(data_path, img_size=(64,64)):
    images = []
    labels = []

    for folder_name in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder_name)
        if os.path.isdir(folder_path):
            label = folder_name
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)
                img = img / 255.0  # Normalize pixel values to between 0 and 1
                images.append(img)
                labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Load and preprocess the dataset
data_path = r"C:\Users\Computer_PA24\Downloads\hand-gesture-recognition-code\Data" # Replace with the path to your dataset
images, labels = load_and_preprocess_data(data_path)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Shuffle and split the dataset
images, labels_encoded = shuffle(images, labels_encoded, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)
print(labels_encoded.shape)
# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    Dropout(0.2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
y_pred = np.argmax(model.predict(x_test), axis=1)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Accuracy:", accuracy_score(y_test, y_pred)*100)
# Save the model
#model.save('hand_gesture_model_weights_new.h5')
