import numpy as np
from neuralnet import nn

best_actions = np.load('/home/henryp/PycharmProjects/SuperMarioBrosRL/best_actions.npy')
best_states = np.load('/home/henryp/PycharmProjects/SuperMarioBrosRL/best_states.npy')

n_outputs = np.max(best_actions)

train_data = best_states[:16000]
train_labels = best_actions[:16000]

test_data = best_states[16000:]
test_labels = best_actions[16000:]

epochs = 200
verbose = 1
batch_size = 1000

nn(n_outputs=n_outputs, train_data=train_data, train_labels=train_labels, test_data=test_data,
   test_labels=test_labels, epochs=epochs, verbose=verbose, batch_size=batch_size)