import tensorflow as tf
import numpy as np

neurons = 0

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0

test_images = test_images / 255.0

test_loss = [0, 0, 0, 0, 0]
train_loss = [0, 0, 0, 0, 0]
test_acc = [0, 0, 0, 0, 0]
train_acc = [0, 0, 0, 0, 0]

for i in range(5):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        #tf.keras.layers.Dense(neurons, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    model.fit(train_images, train_labels, epochs=10)

    test_loss[i], test_acc[i] = model.evaluate(test_images,  test_labels, verbose=2)
    train_loss[i], train_acc[i] = model.evaluate(train_images,  train_labels, verbose=2)

for i in range(5):
    print(test_acc[i])

print()

for i in range(5):
    print(train_acc[i])

print()

for i in range(5):
    print(train_acc[i]/test_acc[i])
