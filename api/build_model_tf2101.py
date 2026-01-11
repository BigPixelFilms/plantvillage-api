import tensorflow as tf

NUM_CLASSES = 3   # change if different

# Build model EXACTLY like training
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(255, 255, 3),   # ✅ CORRECTED
    pooling="avg"
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
])

# Load weights extracted from Keras 3
model.load_weights(
    r"C:/Users/india/anaconda_projects/PlantVillage/models/ModelPatato.weights.h5"
)

# Save TF-2.10 compatible model
model.save(
    r"C:/Users/india/anaconda_projects/PlantVillage/models/ModelPatato_FINAL.h5",
    include_optimizer=False
)

print("✅ FINAL TF-2.10 compatible model created")
