import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# === Load Model ===
model = tf.keras.models.load_model("model/skin_model.h5")

# === Class Mapping (same as during training) ===
class_names = ['melanoma', 'nevus', 'seborrheic_keratosis']

# === Load and Preprocess Image ===
img_path = "./sampleOne.jpg"  # replace with your image
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# === Predict ===
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]

# === Show Result ===
print("Predicted Class:", predicted_class)

plt.imshow(img)
plt.title(f"Prediction: {predicted_class}")
plt.axis("off")
plt.show()
