import tensorflow as tf

# Load broken H5 (from Keras 3)
old_model = tf.keras.models.load_model(
    r"C:/Users/india/anaconda_projects/PlantVillage/models/ModelPatato_tf.h5",
    compile=False
)

# Rebuild input layer (THIS FIXES batch_shape/optional)
inputs = tf.keras.Input(shape=(256, 256, 3), name="image_input")
outputs = old_model(inputs)

fixed_model = tf.keras.Model(inputs, outputs)

# Save TF-pure model
fixed_model.save(
    r"C:/Users/india/anaconda_projects/PlantVillage/models/ModelPatato_FINAL.h5",
    include_optimizer=False
)

print("✅ Final TF-2.10 compatible model saved")
