import tensorflow as tf
from tensorflow import keras

# Load Keras 3 model
model = keras.models.load_model(
    r"C:/Users/india/anaconda_projects/PlantVillage/models/ModelPatato.keras",
    compile=False
)

# Save using TF serializer (important)
tf.keras.models.save_model(
    model,
    r"C:/Users/india/anaconda_projects/PlantVillage/models/ModelPatato_tf.h5",
    include_optimizer=False
)

print("TF-compatible H5 saved successfully")

