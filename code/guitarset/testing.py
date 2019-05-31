import numpy as np 
import torch
import torch.nn.functional as F 
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader as DataLoader
import time
import os
import librosa
import operator
import mir_eval.melody as mel

class Dataset():
	def __init__(self, data, target):
		self.data = data
		self.target = target
	def __getitem__(self, index):
		return self.data[index], self.target[index]
	def __len__(self):
		return len(self.data)

class mix_Dataset():
	def __init__(self, data):
		self.data = data
	def __getitem__(self, index):
		return self.data[index]
	def __len__(self):
		return len(self.data)


class Network(nn.Module):
	def __init__(self, length):
		super(Network, self).__init__()
		self.conv1 = nn.Sequential(
						nn.BatchNorm2d(1),
						nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, stride = 1, padding = 2),
						nn.SELU()
						)
		self.pool1 = nn.MaxPool2d((1,7), return_indices = True)
		self.conv2 = nn.Sequential(
						nn.BatchNorm2d(16),
						nn.Conv2d(in_channels = 16, out_channels = 64, kernel_size = 5, stride = 1, padding = 2),
						nn.SELU()
						)
		self.pool2 = nn.MaxPool2d((1,3), return_indices = True)
		self.conv3 = nn.Sequential(
						nn.BatchNorm2d(64),
						nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, stride = 1, padding = 2),
						nn.SELU()
						)
		self.pool3 = nn.MaxPool2d((1,2), return_indices = True)

		self.bottom = nn.Sequential(

						nn.BatchNorm2d(128),
						nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = (3, 1), stride = 1, padding = (1,0) ),
						nn.SELU(),

						nn.BatchNorm2d(64),
						nn.Conv2d(64, 16, (3,1), 1, (1,0)),
						nn.SELU(),

						nn.BatchNorm2d(16),
						nn.Conv2d(16, 4, (3,1), 1, (1,0)),
						nn.SELU(),

						nn.BatchNorm2d(4),
						nn.Conv2d(4, 1, (3,2), 1, (1,0)),
						nn.SELU()
			)

		self.upool3 = nn.MaxUnpool2d((1,2))
		self.up_conv3 = nn.Sequential(
						nn.BatchNorm2d(128),
						nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 5, stride = 1, padding = 2),
						nn.SELU()
						)
		self.upool2 = nn.MaxUnpool2d((1,3))
		self.up_conv2 = nn.Sequential(
						nn.BatchNorm2d(64),
						nn.Conv2d(in_channels = 64, out_channels = 16, kernel_size = 5, stride = 1, padding = 2),
						nn.SELU()
						)
		self.upool1 = nn.MaxUnpool2d((1,7))
		self.up_conv1 = nn.Sequential(
						nn.BatchNorm2d(16),
						nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size = 5, stride = 1, padding = 2),
						nn.SELU()
						)
		self.rnn = nn.RNN(85,85,8, batch_first = True, bidirectional = True)
		self.softmax = nn.Softmax(dim = 2)
		
		

	def forward(self, x):
		c1, ind1 = self.pool1(self.conv1(x))
		c2, ind2 = self.pool2(self.conv2(c1))
		c3, ind3 = self.pool3(self.conv3(c2))
		bm = self.bottom(c3)
		u3 = self.up_conv3(self.upool3(c3, ind3))
		u2 = self.up_conv2(self.upool2(u3, ind2))
		u1 = self.up_conv1(self.upool1(u2, ind1))
		output, _ = self.rnn(torch.cat((u1, bm), dim = 3)[:,0,:])
		output = self.softmax(output[:,:,:85]+output[:,:,85:])
		return output, bm

	def init_hidden(self, bs): 
		weight = next(self.parameters()).data_nums
		return Variable(wwight.new(8, bs, 85).zero_())

def pos_weight(data):
	rate = 30
	w = 1
	data = np.asarray(data)
	frames = data.shape[-2]
	freq_len = data.shape[-1]
	non_solo = np.sum(data[0,:,-1]) 
	solo = (len(data[0,:])) - non_solo
	z = np.ones((freq_len))
	z[:-1] = non_solo * 200/ (non_solo + solo)
	z[-1] = solo * rate / (non_solo + solo)
	return torch.from_numpy(z).double().cuda()

def mirrval_trans(y_pred, bin_C):
	y_pred = y_pred.view(2153,-1)
	freq = []
	time = librosa.frames_to_time(np.arange(len(y_pred)), sr = 22050, hop_length = 512)
	for frame in y_pred:
		index, value = max(enumerate(frame), key = operator.itemgetter(1))
		if index != 84 :
			freq.append(bin_C[index])
		else:
			freq.append(0)
	return  time, np.asarray(freq)



def testing(test_xlist, test_ylist, mix_list):
	model = Network(2153)
	model.load_state_dict(torch.load( "cnnparams.pkl"))
	model.cuda()
	model.float()
	model.eval()

	bin_C = librosa.core.cqt_frequencies(84, fmin = librosa.note_to_hz('C1'))

	data_test = Dataset(test_xlist, test_ylist)
	data_mix = mix_Dataset(mix_list)
	data_loader = DataLoader(dataset = data_test, batch_size = 1, shuffle = False)
	mix_loader = DataLoader(dataset = data_mix, batch_size = 1, shuffle = True)

	for step, ((batch_x, batch_y),(mix)) in enumerate (zip(data_loader, mix_loader)):
		scores = 0
		x = Variable((batch_x + mix).cuda(), requires_grad = False)
		y = Variable(batch_y.cuda(), requires_grad = False)
		y_pred, bm = model(x.float())
		est_time, est_freq = mirrval_trans(y_pred, bin_C)
		ref_time, ref_freq = mirrval_trans(y, bin_C)
		scores = mel.evaluate(ref_time, ref_freq, est_time, est_freq)
		print(scores['Voicing Recall'],scores['Voicing False Alarm'])

def main(cuda_num):
	test_xlist = []
	test_ylist = []
	mix_list = []

	data_dir = "/home/lego/guitar_solo_detection/feature/guitarset/"

	for n in range(161,181):
		test_xlist.append(np.load(data_dir + "x" + str(n) + ".npy").astype(np.float)[np.newaxis])
		test_ylist.append(np.load(data_dir + "y" + str(n) + ".npy").astype(np.float))
		mix_list.append(np.load(data_dir + "m" + str(n) + ".npy").astype(np.float)[np.newaxis]*0.4)
	with torch.cuda.device(cuda_num):
		testing(test_xlist,test_ylist,mix_list)

cuda_num = 0
main(cuda_num)







