import tensorflow as tf

NUM_CLASSES = 3

# Build base model EXACTLY
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(255, 255, 3),
    pooling="avg"
)

base_model.trainable = False

# 🔑 Functional API (IMPORTANT)
inputs = tf.keras.Input(shape=(255, 255, 3), name="input_image")
x = base_model(inputs)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

model = tf.keras.Model(inputs, outputs)

# 🔑 Load weights by name
model.load_weights(
    r"C:/Users/india/anaconda_projects/PlantVillage/models/ModelPatato_tf.weights.h5",
    by_name=True,
    skip_mismatch=True
)

model.save(
    r"C:/Users/india/anaconda_projects/PlantVillage/models/ModelPatato_FINAL.h5",
    include_optimizer=False
)

print("✅ FINAL TF-2.10 compatible model created")
