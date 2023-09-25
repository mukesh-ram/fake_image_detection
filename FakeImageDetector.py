import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import matplotlib.pyplot as plt

# Specify your directory paths
genuine_data_dir = "C:\\Users\\mukes\\Downloads\\CASIA v2.0\\CASIA2\\Au"
fake_data_dir = "C:\\Users\\mukes\\Downloads\\CASIA v2.0\\CASIA2\\Tp"

# Image dimensions and other parameters
img_height, img_width = 128, 128
batch_size = 32

# Filter supported image formats (JPEG, PNG)
supported_formats = ['.jpg', '.jpeg', '.png']

def filter_supported_images(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                image_files.append(os.path.join(root, file))
    return image_files

# Filter and load genuine images
genuine_image_files = filter_supported_images(genuine_data_dir)

# Convert TIF files to JPEG (if needed)
for tif_file in [file for file in genuine_image_files if file.lower().endswith('.tif')]:
    jpeg_output_path = os.path.splitext(tif_file)[0] + '.jpg'
    image = Image.open(tif_file)
    image.save(jpeg_output_path, 'JPEG')

# Filter and load fake images
fake_image_files = filter_supported_images(fake_data_dir)

# Create DataFrames for genuine and fake images
genuine_df = pd.DataFrame({'filename': genuine_image_files, 'label': 'genuine'})  # Use text labels
fake_df = pd.DataFrame({'filename': fake_image_files, 'label': 'fake'})  # Use text labels

# Concatenate the DataFrames
data_df = pd.concat([genuine_df, fake_df], ignore_index=True)

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split 20% of data for validation
)

# Use flow_from_dataframe to load and preprocess the dataset for genuine images
train_generator = train_datagen.flow_from_dataframe(
    dataframe=data_df,
    x_col='filename',
    y_col='label',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'  # Specify training subset
)

validation_generator = train_datagen.flow_from_dataframe(
    dataframe=data_df,
    x_col='filename',
    y_col='label',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # Specify validation subset
)

# Custom data generator for fake images
def fake_image_data_generator(df, batch_size, img_height, img_width):
    while True:
        fake_samples = df.sample(batch_size)
        x = np.zeros((batch_size, img_height, img_width, 3))
        y = np.zeros(batch_size)
        
        for i, (_, row) in enumerate(fake_samples.iterrows()):
            img = Image.open(row['filename'])
            img = img.resize((img_height, img_width))
            img = np.array(img)
            x[i] = img / 255.0
            y[i] = 1  # Label for fake images
        
        yield x, y

# Create a generator for fake images
fake_generator = fake_image_data_generator(data_df[data_df['label'] == 'fake'], batch_size, img_height, img_width)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model
epochs = 10

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Evaluate the model on fake test data
fake_test_loss, fake_test_acc = model.evaluate(fake_generator, steps=len(fake_image_files) // batch_size)
print(f'Fake Test accuracy: {fake_test_acc * 100:.2f}%')

# Save the trained model
model.save("fake_image_detection_model.h5")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.show()