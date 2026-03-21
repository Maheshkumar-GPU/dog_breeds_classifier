import tensorflow as tf
import numpy as np
import json
from config import model_path, label_map_path, IMAGE_SIZE

model = tf.keras.models.load_model(model_path)

with open(label_map_path) as f:
    label_map = json.load(f)

class_names = list(label_map.keys())

img = tf.keras.preprocessing.image.load_img(
    r"C:\Users\Asus\Downloads\dingo_121.jpg",
    target_size=IMAGE_SIZE
)

img_array = tf.keras.preprocessing.image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]

print("Predicted Breed:", predicted_class)