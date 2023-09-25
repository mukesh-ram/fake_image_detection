# fake_image_detection

FakeImageDetector.py is the code for how to build and train a binary image classification model using TensorFlow and Keras. The goal of this code is to classify images as either "genuine" or "fake." It performs the following steps:

1. **Specify Directory Paths**:
   - `genuine_data_dir`: Path to the directory containing genuine images.
   - `fake_data_dir`: Path to the directory containing fake images.

2. **Image Dimensions and Parameters**:
   - `img_height` and `img_width`: Dimensions to which images are resized.
   - `batch_size`: The batch size used during training.

3. **Filter Supported Image Formats**:
   - The `filter_supported_images` function is defined to filter image files in the specified directories based on supported formats (JPEG, PNG). It walks through the directory tree and collects the paths of image files with valid extensions.

4. **Preprocess Genuine Images**:
   - Genuine images are filtered and loaded from the `genuine_data_dir`.
   - Any TIFF (TIF) images are converted to JPEG format for compatibility.

5. **Preprocess Fake Images**:
   - Fake images are filtered and loaded from the `fake_data_dir`.

6. **Create DataFrames for Images**:
   - Two Pandas DataFrames are created to store information about genuine and fake images.
   - `genuine_df` and `fake_df` contain columns for the image filename and the label ("genuine" or "fake").
   - Data from both DataFrames is concatenated into a single `data_df`.

7. **Data Augmentation for Training**:
   - An `ImageDataGenerator` is configured for data augmentation during training. Augmentation techniques include rotation, width and height shifts, and horizontal flipping.
   - The data is split into training and validation sets using `validation_split`.

8. **Load and Preprocess Data for Genuine Images**:
   - Two data generators (`train_generator` and `validation_generator`) are created using `flow_from_dataframe`.
   - These generators load and preprocess images from the DataFrame, specifying training and validation subsets.

9. **Custom Data Generator for Fake Images**:
   - A custom data generator named `fake_image_data_generator` is defined to handle fake images. This generator randomly selects a batch of fake images, preprocesses them, and assigns a label of 1 (indicating "fake").

10. **Define the Model**:
    - A Convolutional Neural Network (CNN) model is defined using Keras' Sequential API.
    - It consists of several convolutional layers, max-pooling layers, and fully connected layers.
    - The model is compiled with the Adam optimizer and binary cross-entropy loss for binary classification.

11. **Model Training**:
    - The model is trained using the `fit` method, with training and validation data provided by the generators.
    - Training occurs for a specified number of epochs.

12. **Evaluate on Fake Test Data**:
    - The trained model is evaluated on a separate set of fake test data.
    - A custom data generator (`fake_generator`) is used to load and preprocess the fake test images.

13. **Save the Trained Model**:
    - The trained model is saved to a file named "fake_image_detection_model.h5."

This code provides a complete pipeline for training an image classification model to distinguish between genuine and fake images. It includes data preprocessing, model definition, training, and evaluation.


#detect.py
detect.py code is a Python script that demonstrates how to use a trained deep learning model to classify images as either "Real" or "Fake." The primary components of this code include loading a pre-trained model, defining image preprocessing functions, classifying images, and optionally generating and classifying GAN-generated images.

Here's a detailed breakdown of the code:

1. **Loading the Pre-Trained Model**:
   - The code starts by loading a pre-trained fake image detection model using TensorFlow's `load_model` function. The model is loaded from a file named `'fake_image_detection_model.h5'`, which should contain a trained model for binary image classification.

2. **Defining Image Dimensions and Preprocessing Functions**:
   - The script defines the dimensions `img_height` and `img_width` for resizing input images. These dimensions should match the requirements of the loaded model.
   - Two functions are defined:
     - `preprocess_image(image_path)`: This function takes an image file path as input, opens the image using the PIL library, resizes it to the specified dimensions, and normalizes the pixel values to the range [0, 1]. The preprocessed image is returned as a NumPy array.
     - `classify_image(image_path)`: Given an image file path, this function preprocesses the image using `preprocess_image` and then classifies it using the loaded model. If the model predicts a probability greater than 0.5, the image is classified as "Fake"; otherwise, it's classified as "Real."

3. **Example Usage**:
   - Two examples of using the defined functions are provided:
     - **Classify an Uploaded Image**:
       - `uploaded_image_path` is set to the path of an uploaded image (you should replace this with the actual path of your uploaded image).
       - The `classify_image` function is called with the path of the uploaded image, and the result is printed. This demonstrates how to classify an image provided by the user.



