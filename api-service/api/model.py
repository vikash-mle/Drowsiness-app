import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow_hub as hub
import cv2
import matplotlib.image as mpimg


AUTOTUNE = tf.data.experimental.AUTOTUNE
local_experiments_path = "/persistent/experiments"
best_model = None
best_model_id = None
prediction_model = None
data_details = None
image_width = 145
image_height = 145
num_channels = 3


def load_prediction_model():
    print("Loading Model...")
    global prediction_model, data_details

    best_model_path = "/persistent/experiments/cropping_face_with_aug.h5"

    print("best_model_path:", best_model_path)
    prediction_model = tf.keras.models.load_model(
        best_model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    print(prediction_model.summary())


def load_preprocess_image_from_path(image_path):
    print("Image", image_path)

    image_width = 145
    image_height = 145
    num_channels = 3

    # Prepare the data
    '''
    def load_image(path):
        image = mpimg.imread(path)
        
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=num_channels)
        image = tf.image.resize(image, [image_height, image_width])
        
        return image
    '''
    # Crop Image
    def crop_image(image_path):
        
        IMG_SIZE = 145
        crop = []
        image_array = mpimg.imread(image_path)
        #image_array = np.array(tf.image.decode_jpeg(image, channels=num_channels))
        #image = tf.image.resize(image, [image_height, image_width])       
        #image_array = cv2.imread(image, cv2.IMREAD_COLOR)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml" )
        faces = face_cascade.detectMultiScale(image_array, 1.01, 3)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(image_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_color = img[y:y+h, x:x+w]
            resized_array = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
            crop.append([resized_array])
        return crop

    # Normalize pixels
    def normalize(image):
        image = image / 255
        return image

    test_data = tf.data.Dataset.from_tensor_slices(crop_image(image_path)[0])
    #test_data = test_data.map(crop_image, num_parallel_calls=AUTOTUNE)
    test_data = test_data.map(normalize, num_parallel_calls=AUTOTUNE)
    test_data = test_data.repeat(1).batch(1)

    return test_data


def make_prediction(image_path):
    load_prediction_model()

    # Load & preprocess
    test_data = load_preprocess_image_from_path(image_path)

    # Make prediction
    prediction = prediction_model.predict(test_data)
    idx = np.argmax(prediction)
    data_details = ['Closed', 'Open' ,'no_yawn', 'yawn']
    prediction_label = data_details[idx]

    if prediction_model.layers[-1].activation.__name__ != 'softmax':
        prediction = tf.nn.softmax(prediction).numpy()
        print(prediction)

    sleepy = False
    if prediction_label == "Closed" or "yawn":
        sleepy = True

    return {
        "input_image_shape": str(test_data.element_spec.shape),
        "prediction_shape": prediction.shape,
        "prediction_label": prediction_label,
        "prediction": prediction.tolist(),
        "accuracy": round(np.max(prediction)*100, 2),
        "sleepy": sleepy
    }
