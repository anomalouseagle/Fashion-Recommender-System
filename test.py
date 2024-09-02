import pickle
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2
import os

# Define the ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Create a Sequential model and add the ResNet50 base model and GlobalMaxPooling2D layer
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Load the feature list and filenames
try:
    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit()

# Load and preprocess the image
try:
    img = image.load_img('sample/jersey.jpg', target_size=(224, 224))
except FileNotFoundError as e:
    print(f"Image not found: {e}")
    exit()

img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)

# Get the prediction and normalize the result
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

# Fit the NearestNeighbors model
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

# Find the nearest neighbors
distances, indices = neighbors.kneighbors([feature_list])
print(indices)

# Display the nearest images
for file in indices[0]:
    temp_img_path = filenames[file]
    temp_img = cv2.imread(temp_img_path)
    
    if temp_img is None:
        print(f"Error loading image: {temp_img_path}")
        continue

    cv2.imshow('output', temp_img)
    resized_img = cv2.resize(temp_img, (224, 224))
    cv2.imshow('resized_output', resized_img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
