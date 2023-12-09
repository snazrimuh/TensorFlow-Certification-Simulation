# =============================================================================
# PROBLEM C2
#
# Create a classifier for the MNIST Handwritten digit dataset.
# The test will expect it to classify 10 classes.
#
# Don't use lambda layers in your model.
#
# Syah Rizan Nazri Muhammad
# Sebelas Maret University
#
# Desired accuracy AND validation_accuracy > 91%
# =============================================================================

import tensorflow as tf


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        accuracy = logs.get('accuracy') or 0.0
        val_accuracy = logs.get('val_accuracy') or 0.0

        if accuracy > 0.92 and val_accuracy > 0.92:
            print("\nTraining stopped as it meets the target.")
            self.model.stop_training = True



def solution_C2():
    mnist = tf.keras.datasets.mnist

    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    callback = CustomCallback()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(training_images,
              training_labels,
              validation_data=(test_images, test_labels),
              epochs=10,
              verbose=1)

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C2()
    model.save("model_C2.h5")
