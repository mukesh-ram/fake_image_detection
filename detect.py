import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained fake image detection model
model = load_model("C:\\Users\\mukes\\OneDrive\\Documents\\Moonraft\\fake_image_detection_model.h5")

# Define image dimensions
img_height, img_width = 128, 128

# Function to preprocess an uploaded image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((img_height, img_width))
    img = np.array(img)
    img = img / 255.0  # Normalize the image
    return img

# Function to classify an image as real or fake
def classify_image(image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    result = model.predict(img)
    if result[0] > 0.5:
        return "Fake"
    else:
        return "Real"

# Specify the path to the image you want to classify
image_path = "C:\\Users\\mukes\\Downloads\\CASIA v2.0\\CASIA2\\Au\\Au_sec_30390.jpg"  # Replace with the actual image path

# Classify the image
classification_result = classify_image(image_path)
print(f"The uploaded image is {classification_result}")
