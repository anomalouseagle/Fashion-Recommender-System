import os
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm

# Load the ResNet50 model without the top layer
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False


# Add GlobalMaxPooling2D layer to the model
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Print the model summary
print(model.summary())


# Define a function to extract features
def extract_features(img_path, model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis = 0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalised_result = result / norm(result)

    return normalised_result

filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

feature_list = []

for file in filenames:
    feature_list.append(extract_features(file,model))

print(len(filenames))
print(filenames[0:5])
