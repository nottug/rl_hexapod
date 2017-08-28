import math
import os
import sys
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from actor_critic import AC
from cnn import CNN
from servo import Servo
#from bno055 import BNO055

path = '/home/ubuntu/hexapod/code/models/model.pth'

pi = Variable(torch.FloatTensor([[math.pi]]))
torch.manual_seed(1)

width = 600
height = 480
thresh = 0.4

hex = Servo(245)
darknet = CNN(thresh, width, height)

'''
bno = BNO055()
if bno.begin() is not True:
	print('error with bno')
	exit()
'''

state_max = 2400
max_deg = 90

w_time = 1

gamma = 0.99
tau = 1

x, y, z = 0, 0, 0
a_x, a_y, a_z = 0, 0, 0
first_ob = True
last_time = time.time()


def normal(x, mu, sigma_sq):
	a = (-1 * (Variable(x) - mu).pow(2) / (2 * sigma_sq)).exp()
	b = 1 / (2 * sigma_sq * pi.expand_as(sigma_sq)).sqrt()
	return a * b


def find_reward(obs):
	area = obs[0][1]
	dist = abs(obs[0][0])

	return -(dist*0.5) + area


def retrieve_state(action):
	vision = torch.from_numpy(darknet.process())

	'''
	if not first_ob:
		diff = time.time() - last_time
		x += (a_x * diff) * (180 / math.pi)
		y += (a_y * diff) * (180 / math.pi)
		z += (a_z * diff) * (180 / math.pi)
	else:
		first_ob = False
	last_time = time.time()
	print('got time')
	a_x, a_y, a_z = bno.getVector(0x14)
	print('got gyro')
	gyro = np.array([a_x, a_y]).astype('float32')
	print('return obs')
	'''

	return torch.cat([torch.from_numpy(action/state_max).transpose(0,1), vision.transpose(0,1)]).transpose(0,1)


def clip_grad_norm(parameters, max_norm, norm_type=2):
	parameters = list(filter(lambda p: p.grad is not None, parameters))
	max_norm = float(max_norm)
	norm_type = float(norm_type)
	if norm_type == float('inf'):
  		total_norm = max(p.grad.data.abs().max() for p in parameters)
	else:
		total_norm = 0
		for p in parameters:
			param_norm = p.grad.data.norm(norm_type)
			total_norm += param_norm ** norm_type
		total_norm = total_norm ** (1. / norm_type)
	clip_coef = max_norm / (total_norm + 1e-6)
	if clip_coef < 1:
		for p in parameters:
			p.grad.data.mul_(clip_coef)
	return total_norm


def train(epochs, episodes, wait_time, lr):
	model = AC()
	model.load_state_dict(torch.load(path))
	optimizer = optim.Adam(model.parameters(), lr=lr)
	model.train()

	try:
		for epoch in range(epochs):
			values = []
			log_probs = []
			rewards = []
			entropies = []

			action = hex.reset()
			time.sleep(w_time)

			for episode in range(episodes):
				print("episode {}".format(episode))
				state = retrieve_state(action)
				mu, sigma, value = model(Variable(state))

				sigma = F.softplus(sigma)
				eps = torch.randn(mu.size())
				action = (mu + sigma.sqrt() * Variable(eps)).data
				prob = normal(action, mu, sigma)

				entropy = -0.5 * ((sigma + 2 * pi.expand_as(sigma)).log() + 1)
				entropies.append(entropy)
				log_prob = prob.log()

				action = action.numpy()
				hex.apply(action * state_max)
				time.sleep(wait_time)

				reward = find_reward(darknet.process())
				values.append(value)
				log_probs.append(log_prob)
				rewards.append(reward)

			state = retrieve_state(action)
			_, _, value = model(Variable(state))
			R = value.data
			values.append(Variable(R))

			policy_loss = 0
			value_loss = 0

			R = Variable(R)
			gae = torch.zeros(1, 1)

			for i in reversed(range(len(rewards))):
				R = gamma * R + rewards[i]
				advantage = R - values[i]
		   	 	value_loss = value_loss + 0.5 * advantage.pow(2)
				delta_t = rewards[i] + gamma * values[i + 1].data - values[i].data
				gae = gae * gamma * tau + delta_t
				policy_loss = policy_loss - (log_probs[i] * Variable(gae).expand_as(log_probs[i])).sum() - (0.01 * entropies[i]).sum()
			
			print(policy_loss)
			optimizer.zero_grad()
			(policy_loss.unsqueeze(0) + 0.5 * value_loss).backward()
			clip_grad_norm(model.parameters(), 40)
			optimizer.step()
			
			torch.save(model.state_dict(), path)
		hex.reset()
	except:
		hex.reset()
		torch.save(model.state_dict(), path)
