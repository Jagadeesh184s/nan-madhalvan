import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data
# Normalize pixel values to be between 0 and 1
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255

# Reshape images to include channel dimension (28x28x1)
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

# Convert labels to categorical one-hot encoding
num_classes = 10
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

# Build the model
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])

model.summary()

# Compile the model
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Train the model
batch_size = 128
epochs = 15
history = model.fit(
    train_images, train_labels,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1
)

# Evaluate on test set
score = model.evaluate(test_images, test_labels, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Save the model
model.save("mnist_cnn.h5")

# Make a prediction on a single image
def predict_digit(image):
    # Preprocess the image (same as training data)
    image = image.astype("float32") / 255
    image = np.expand_dims(image, -1)
    image = np.expand_dims(image, 0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)
    return np.argmax(prediction)

# Example usage with a test image
sample_idx = 0  # Change this to test different images
sample_image = test_images[sample_idx]
sample_label = np.argmax(test_labels[sample_idx])

# Reshape for display (remove channel dimension)
display_image = np.squeeze(sample_image, axis=-1)

plt.imshow(display_image, cmap="gray")
plt.title(f"True: {sample_label}, Predicted: {predict_digit(sample_image)}")
plt.show()
