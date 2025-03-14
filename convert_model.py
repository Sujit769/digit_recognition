import tensorflow as tf

# Load the existing model (.h5 format)
model = tf.keras.models.load_model("digit_recognition_model.h5")

# Save it in the new .keras format
model.save("digit_recognition_model.keras", save_format="keras")

print("Model has been successfully converted to .keras format.")