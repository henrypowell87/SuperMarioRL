import numpy as np
import torch
from torch.nn import Softmax

device = torch.device('cuda:0')

# Generate batch of episodes
def generate_batch(env, batch_size, net, render=False):
    activation = Softmax(dim=1)

    batch_actions, batch_states, batch_rewards = [], [], []

    life = 2
    for i in range(batch_size):
        print('Trial: ' + str(i+1))
        states, actions = [], []
        total_reward = 0
        s = env.reset()
        for step in range(5000):
            s_v = torch.FloatTensor([s])
            s_v = s_v.permute(0, 3, 1, 2)
            s_v.to(device)
            act_probs_v = activation(net(s_v))
            act_probs_v = torch.Tensor.cpu(act_probs_v)
            act_probs = act_probs_v.data.numpy()[0]

            a = np.random.choice(len(act_probs), p=act_probs)

            new_s, r, done, info = env.step(a)

            states.append(s)
            actions.append(a)
            total_reward += r

            s = new_s

            if render:
                env.render()

            if done or info['life'] < life or info['time'] == 0 or info['flag_get']:
                batch_actions.append(actions)
                batch_states.append(states)
                batch_rewards.append(total_reward)
                break
    return batch_actions, batch_states, batch_rewards


# Return best states and actions
def filter_batch(batch_actions, batch_states, batch_rewards, percentile=80):

    threshold = np.percentile(batch_rewards, percentile)

    best_actions, best_states = [], []

    for i in range(len(batch_rewards)):
        if batch_rewards[i] >= threshold:
            for j in range(len(batch_states[i])):
                best_actions.append(batch_actions[i][j])
                best_states.append(batch_states[i][j])

    return best_actions, best_states


def play_game(env, net, rounds_to_play=1):

    activation = Softmax(dim=1)
    life = 2
    for i in range(rounds_to_play):
        print('Attempt: ' + str(i+1))
        s = env.reset()
        for step in range(5000):
            s_v = torch.FloatTensor([s])
            s_v = s_v.to(device)
            act_probs_v = activation(net(s_v))
            act_probs_v = torch.Tensor.cpu(act_probs_v)
            act_probs = act_probs_v.data.numpy()[0]

            a = np.random.choice(len(act_probs), p=act_probs)

            new_s, r, done, info = env.step(a)

            s = new_s

            env.render()

            if done or info['life'] < life or info['time'] == 0 or info['flag_get']:
                break
