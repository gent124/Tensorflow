import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


print("Train data shape:", x_train.shape)
print("Test data shape:", x_test.shape)


x_train, x_test = x_train / 255.0, x_test / 255.0


print("Train data after normalization:", x_train[0])


model = models.Sequential(
    [
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax"),
    ]
)


model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


model.fit(x_train, y_train, epochs=5)


test_loss, test_acc = model.evaluate(x_test, y_test)
print("\nTest accuracy:", test_acc)


history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()
