import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader as DataLoader
import numpy as np 
import time
import os

class Dataset():
	def __init__(self, data, target):
		self.data = data
		self.target = target
	def __getitem__(self, index):
		return self.data[index], self.target[index]
	def __len__ (self):
		return len(self.data)

class network(nn.Module):
	def __init__(self, input_size, seq_len):
		super(network, self).__init__()
		self.seq_len = seq_len
		self.conv1 = nn.Sequential(
			nn.BatchNorm2d(1),
			nn.Conv2d(in_channels=1, out_channels=32, kernel_size=9, stride=1, padding=4),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2,4)),
		)
		self.conv2 = nn.Sequential(
			nn.BatchNorm2d(32),
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2,4)),
		)
		self.conv3 = nn.Sequential(
			nn.BatchNorm2d(64),
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2,4)),
		)
		self.conv4 = nn.Sequential(
			nn.BatchNorm2d(128),
			nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2,2)),
		)
		self.drop = nn.Dropout(p=0.4)
		self.linear = nn.Linear(13 ,1)
		self.softmax = nn.Softmax()

	def forward(self, x):
		x = x[:,np.newaxis,:,:]
		h = self.conv1(x)
		h = self.drop(h)
		r = self.conv2(h)
		r = self.drop(r)
		r = self.conv3(r)
		r = self.drop(r)
		r = self.conv4(r)
		r = self.drop(r)
		output = self.linear(r.view(r.size()[0],13))
		#output = self.softmax(output)
		return output.view(output.size(0))

def train(train_xlist, train_ylist, val_xlist, val_ylist, epoch_num=1000, lr=0.0001):
	model = network(168,215)
	model.double()
	model.cuda()
	opt = optim.Adam(model.parameters(), lr=lr)

	min_loss =1000


	for epoch in range (epoch_num):
		tp, fp, fn, tn, f1= 0, 0, 0, 0, 0
		startime = time.time()

		#train
		train_loss = 0
		model.train()

		data_train = Dataset(train_xlist,train_ylist)
		train_loader = DataLoader(dataset = data_train, batch_size = 5, shuffle = True )

		"""
		ind = np.arange(len(train_xlist))
		for i in ind:
			x = train_xlist[i][np.newaxis,:,:]
			y = train_ylist[i]
			x = Variable(torch.from_numpy(x).cuda(), volatile=True)
			y = Variable(torch.from_numpy(y).cuda(), volatile=True)
		"""
		for step, (batch_x, batch_y) in enumerate(train_loader):
			x = Variable(batch_x.cuda(), requires_grad=True)
			y = Variable(batch_y.view(batch_y.size(0)).cuda(), requires_grad=True)

			y_pred = model(x)
			#print (F.sigmoid(y_pred)[0].item(),y[0].item(),"||",F.sigmoid(y_pred)[1].item(),y[1].item())

			loss_fn = F.binary_cross_entropy_with_logits(y_pred, y)
			opt.zero_grad()
			loss_fn.backward()
			opt.step()
			train_loss += loss_fn.data[0]
		train_loss /= step

		#val
		val_loss = 0
		model.eval()

		data_val = Dataset(val_xlist,val_ylist)
		val_loader = DataLoader(dataset = data_val, batch_size = 5, shuffle = False)

		for step, (batch_x, batch_y) in enumerate(val_loader):
			x = Variable(batch_x.cuda())
			y = Variable(batch_y.view(batch_y.size(0)).cuda())
			y_pred = model(x)
			loss_fn = F.binary_cross_entropy_with_logits(y_pred,y)
			val_loss += loss_fn.data[0]
			#print (F.sigmoid(y_pred)[0].item(),y[0].item(),"||",F.sigmoid(y_pred)[1].item(),y[1].item())
			y_pred = model(x).cpu().data.numpy()
			if y_pred[0] > 0.5:
				if val_ylist[step-1] == 1:
					tp += 1
				elif val_ylist[step-1] == 0:
					fp += 1
				else:
					print ('f1 positive error')
			else:
				if val_ylist[step-1] == 1:
					fn += 1
				elif val_ylist[step-1] == 0:
					tn += 1
				else:
					print ('f1 negative error')
		val_loss /= step
		f1 = float(2*tp)/(2*tp+fn+fp)

		print("epoch: ", epoch ," | train_loss: " ,"%.4f" % train_loss, " | val_loss: " ,"%.4f" % val_loss, " | f1: %.4f " % f1, " | time = " ,int(time.time()-startime) )

		if val_loss < min_loss:
			min_loss = val_loss
			min_epoch = epoch
			print("min epoch: " ,min_epoch)
		torch.save(model.state_dict(), 'cnnparams.pkl')

def main(cuda_num):

	train_xlist = []
	train_ylist = []
	val_xlist = []
	val_ylist = []

	data_dir = "/home/lego/guitar_solo_detection/feature/mel_seg/"

	for j in range (1000):

		x_path = data_dir + "train/" + str(0) + "/x" + str(j+1) + ".npy"
		y_path = data_dir + "train/" + str(0) + "/y" + str(j+1) + ".npy"
		x = abs(np.load(x_path))
		y = abs(np.load(y_path))
		train_xlist.append(np.array(x, dtype=float))
		train_ylist.append(np.array(y, dtype=float))

	for j in range (600):

		x_path = data_dir + "train/" + str(1) + "/x" + str(j+1) + ".npy"
		y_path = data_dir + "train/" + str(1) + "/y" + str(j+1) + ".npy"
		x = abs(np.load(x_path))
		y = abs(np.load(y_path))
		train_xlist.append(np.array(x, dtype=float))
		train_ylist.append(np.array(y, dtype=float))


	for j in range (300):
		
		x_path = data_dir + "val/" + str(0) + "/x" + str(j+1) + ".npy"
		y_path = data_dir + "val/" + str(0) + "/y" + str(j+1) + ".npy"
		x = abs(np.load(x_path))
		y = abs(np.load(y_path))
		val_xlist.append(np.array(x, dtype=float))
		val_ylist.append(np.array(y, dtype=float))

	for j in range (141):
		x_path = data_dir + "val/" + str(1) + "/x" + str(j+1) + ".npy"
		y_path = data_dir + "val/" + str(1) + "/y" + str(j+1) + ".npy"
		x = abs(np.load(x_path))
		y = abs(np.load(y_path))
		val_xlist.append(np.array(x, dtype=float))
		val_ylist.append(np.array(y, dtype=float))

	with torch.cuda.device(cuda_num):
		train(train_xlist, train_ylist, val_xlist, val_ylist)

cuda_num = 0
main(cuda_num)




