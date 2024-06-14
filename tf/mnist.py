import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tracemalloc

tf.random.set_seed(1)
np.random.seed(1)

def convert_to_gigabye(byte_tuple):
    gigabyte_values = [bytes_value / 1_073_741_824 for bytes_value in byte_tuple]
    return gigabyte_values

import time
class etimer(tf.keras.callbacks.Callback):
    def __init__ (self): # initialization of the callback
        super(etimer, self).__init__()
    def on_epoch_begin(self, epoch, logs=None):
        self.now= time.time()
        tracemalloc.start()
    def on_epoch_end(self,epoch, logs=None): 
        later=time.time()
        duration=later-self.now 
        print('\nfor epoch ', epoch +1, ' the duration was ', duration, ' seconds')
        print('\nmemory usage: ', convert_to_gigabye(tracemalloc.get_traced_memory()), ' GiB')
        tracemalloc.stop()

# Load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train, X_validation = X_train[:50000], X_train[50000:]
Y_train, Y_validation = Y_train[:50000], Y_train[50000:]

# normalize
X_train = X_train / 255.0
X_validation = X_validation / 255.0
X_test = X_test / 255.0

Y_train = to_categorical(Y_train, 10)
Y_validation = to_categorical(Y_validation, 10)
Y_test = to_categorical(Y_test, 10)

image_size = X_train.shape[1]
X_train = np.reshape(X_train, [-1, image_size * image_size])
X_test = np.reshape(X_test, [-1, image_size * image_size])
X_validation = np.reshape(X_validation, [-1, image_size * image_size])

# Set initial parameters
rnn_out_size = 64
seq_len = 4
in_size = 196
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 15e-3

X_train = X_train.reshape(-1, seq_len, in_size)
X_validation = X_validation.reshape(-1, seq_len, in_size)
X_test = X_test.reshape(-1, seq_len, in_size)

# Create the RNN model using Keras
def RNN_model(rnn_out_size, num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(seq_len, in_size)))
    model.add(tf.keras.layers.SimpleRNN(rnn_out_size))
    model.add(tf.keras.layers.Dense(num_classes, activation=None))
    return model

model = RNN_model(rnn_out_size, num_classes)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), epochs=num_epochs, batch_size=batch_size, callbacks=[etimer()])

# Evaluate the model
train_loss, train_acc = model.evaluate(X_train, Y_train, batch_size=batch_size)
test_loss, test_acc = model.evaluate(X_test, Y_test, batch_size=batch_size)

print('\nTrain Accuracy: {:.4f}   Test Accuracy: {:.4f}   Train Loss: {:.4f}   Test Loss: {:.4f}'.format(train_acc, test_acc, train_loss, test_loss))
