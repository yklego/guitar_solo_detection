import numpy as np 
import torch.nn as nn
import torch 
import torch.nn.functional as F 
from torch.autograd import Variable
from torch.utils.data import DataLoader as DataLoader
import time
import os

class Dataset():
	def __init__(self, data, target);
		self.data = data
		self.target = target
	def __getitem__(self, index):
		return self.data[index], self.target[index]
	def __len__ (self):
		return len(self.data)

class Network(nn.Module):
	def __init__(self, input_size):
		super(Network, self).__init__()
		self.conv1 = nn.Sequential(
						nn.BatchNorm2d(input_size);
						nn.Conv2d(in_channel = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
						nn.MaxPool2d(2)
						nn.ReLU()
						)
		self.conv2 = nn.Sequential(
						nn.BatchNorm2d(input_size);
						nn.Conv2d(in_channel = 16, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
						nn.MaxPool2d(2)
						nn.ReLU()
						)
		self.conv3 = nn.Sequential(
						nn.BatchNorm2d(input_size);
						nn.Conv2d(in_channel = 64, out_channels =128, kernel_size = 3, stride = 1, padding = 1),
						nn.MaxPool2d(2)
						nn.ReLU()
						)
