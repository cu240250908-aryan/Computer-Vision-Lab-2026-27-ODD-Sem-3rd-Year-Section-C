import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, classification_report

os.makedirs("output", exist_ok=True)

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    x_train, y_train,
    epochs=3,
    validation_split=0.1
)

# Test
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("Test Accuracy:", round(acc * 100, 2), "%")

# Graph
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.legend()
plt.savefig("output/accuracy.png")
plt.close()

# Confusion Matrix
pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, pred)

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.savefig("output/confusion_matrix.png")
plt.close()

# Classification Report
with open("output/report.txt", "w") as f:
    f.write(classification_report(y_test, pred))

# Your own image
if os.path.exists("image10.png"):
    img = cv2.imread("image10.png", 0)
    img = cv2.resize(img, (28, 28))

    if np.mean(img) > 127:
        img = 255 - img

    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    result = model.predict(img, verbose=0)
    digit = np.argmax(result)

    print("Your Image Prediction:", digit)

    plt.imshow(img[0].reshape(28,28), cmap="gray")
    plt.title(f"Predicted Digit: {digit}")
    plt.axis("off")
    plt.savefig("output/my_digit_prediction.png")
    plt.close()

print("All outputs saved in output folder!")