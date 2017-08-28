import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def normalized_columns_initializer(weights, std=1.0):
	out = torch.randn(weights.size())
	out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
	return out


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		weight_shape = list(m.weight.data.size())
		fan_in = weight_shape[1]
		fan_out = weight_shape[0]
		w_bound = np.sqrt(6. / (fan_in + fan_out))
		m.weight.data.uniform_(-w_bound, w_bound)
		m.bias.data.fill_(0)


class AC(nn.Module):
	def __init__(self):
		super(AC, self).__init__()
		self.l1 = nn.Linear(20, 64)
		self.l2 = nn.Linear(64, 256)
		# actor
		self.mu_net = nn.Linear(256, 18)
		self.sigma_net = nn.Linear(256, 18)
		# critic
		self.value_net = nn.Linear(256, 1)

		# init weight
		self.apply(weights_init)
		self.mu_net.weight.data = normalized_columns_initializer(
					self.mu_net.weight.data, 0.01)
		self.sigma_net.weight.data = normalized_columns_initializer(
					self.sigma_net.weight.data, 0.01)
		self.mu_net.bias.data.fill_(0)
		self.sigma_net.bias.data.fill_(0) 

		self.value_net.weight.data = normalized_columns_initializer(
						self.value_net.weight.data, 1.0)
		self.value_net.bias.data.fill_(0)

		self.train()


	def forward(self, x):
		x1 = F.relu(self.l1(x))
		x2 = F.relu(self.l2(x1))

		mu = self.mu_net(x2)
		sigma = self.sigma_net(x2)
		value = self.value_net(x2)

		return mu, sigma, value
