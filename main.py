import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from neuralnet import Net
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym_super_mario_bros
from functions import generate_batch, filter_batch

PATH = './MarioRL.pth'

SPEED_RUN_MOVEMENT = [
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A']
]

SIMPLE_MOVEMENT = [
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
]

movement = SPEED_RUN_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, movement)

device = torch.device('cuda:0')

n_outputs = len(movement)
obs_size = 240 * 256 * 3
epochs = 50
batch_size = 20

net = Net(obs_size=obs_size, hidden_size=200, n_actions=n_outputs)

loss = CrossEntropyLoss()
optimizer = optim.Adam(params=net.parameters(), lr=0.001)

trial = 1

for i in range(epochs):
    print('Epoch: ' + str(i+1))
    batch_actions, batch_states, batch_rewards = generate_batch(env=env, batch_size=batch_size, net=net, render=False)

    elite_actions, elite_states = filter_batch(batch_actions=batch_actions, batch_states=batch_states,
                                               batch_rewards=batch_rewards, percentile=80)

    net.cuda()
    elite_states, elite_actions = torch.FloatTensor(elite_states), torch.LongTensor(elite_actions)
    elite_states = elite_states.permute(0, 3, 1, 2)
    elite_states, elite_actions = elite_states.to(device), elite_actions.to(device)

    optimizer.zero_grad()

    outputs = net(elite_states)

    loss_v = loss(outputs, elite_actions)

    loss_v.backward()
    optimizer.step()

    mean_reward, threshold = np.mean(batch_rewards), np.percentile(batch_rewards, 80)
    print("%d: loss=%.3f, reward mean=%.1f, reward threshold=%.1f" % (i, loss_v.item(), mean_reward, threshold))

    torch.save(net.state_dict(), PATH)

torch.save(net.state_dict(), PATH)

net.load_state_dict(torch.load(PATH))

generate_batch(env=env, batch_size=1, net=net, render=True)
env.close()

# Make a function for playing the game through once using the trainined neural net (this will look something
# like generate batch only it doesn't need to store the variables.
