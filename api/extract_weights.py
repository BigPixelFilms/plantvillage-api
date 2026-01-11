import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model(
    r"C:/Users/india/anaconda_projects/PlantVillage/models/ModelPatato.keras",
    compile=False
)

# ✅ CORRECT WAY
model.save_weights(
    r"C:/Users/india/anaconda_projects/PlantVillage/models/ModelPatato_tf.weights.h5"
)

print("✅ TensorFlow-compatible weights saved")
