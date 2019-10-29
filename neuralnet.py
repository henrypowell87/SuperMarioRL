import tensorflow as tf
from tensorflow import keras


def nn(n_outputs, train_data, train_labels, test_data, test_labels, epochs, verbose, batch_size):
    model = keras.Sequential()
    model.add(keras.layers.Dense(100, input_shape=(10000, 240, 256, 3), activation=tf.nn.relu))
    model.add(keras.layers.Dense(n_outputs, activation=tf.nn.sigmoid))

    model.summary()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    history = model.fit(train_data,
                        train_labels,
                        epochs = epochs,
                        #validation_data=(val_data, val_labels),
                        verbose=verbose)

    _, accuracy = model.evaluate(test_data,
                                 test_labels,
                                 batch_size=batch_size,
                                 verbose=verbose)

    return history, _, accuracy