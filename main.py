import numpy as np
from neuralnet import nn
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

# Generate batch of episodes
def generate_batch(batch_size):

    batch_actions, batch_states, batch_rewards = [], [], []

    for i in range(batch_size):

        done = True

        actions, states = [], []
        total_reward = 0

        for step in range(5000):

            if done:
                state = env.reset()

            action = env.action_space.sample()

            state, reward, done, info = env.step(action)

            total_reward += reward

            actions.append(action)

            states.append(state)

            env.render()

        batch_actions.append(actions)
        batch_states.append(states)
        batch_rewards.append(total_reward)

    return batch_actions, batch_states, batch_rewards


#Return best srtates and actions
def filter_batch(batch_actions, batch_states, batch_rewards, percentile=50):

        threshold = np.percentile(batch_rewards, percentile)

        best_actions, best_states = [], []

        for i in range(len(batch_rewards)):
            if batch_rewards[i] > threshold:
                for j in range(len(batch_states[i])):
                    best_actions.append(batch_actions[i][j])
                    best_states.append(batch_states[i][j])

        return best_actions, best_states


batch_actions, batch_states, batch_rewards = generate_batch(10)
best_actions, best_states = filter_batch(batch_actions, batch_states, batch_rewards, percentile=80)

np.save('best_actions', best_actions)
np.save('best_states', best_states)

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

print(best_actions)
print(best_states)
print(np.max(best_actions))

env.close()