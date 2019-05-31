import numpy as np 
import torch
import torch.nn.functional as F 
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader as DataLoader
import time
import os

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
	"""z[:-1] = ((len(data) * frames) * rate / solo)
	z[-1] = ((len(data) * frames) * rate * w / non_solo)"""
	return torch.from_numpy(z).double().cuda()


def train(train_xlist, train_ylist, val_xlist, val_ylist, mtrain_xlist, mval_xlist, epoch_num = 1000, lr = 0.001):
	model = Network(2153)
	model.double()
	model.cuda()
	opt = optim.Adam(model.parameters(), lr=lr)
	min_loss = 1000
	train_loss_weight = pos_weight(train_ylist)
	val_loss_weight = pos_weight(val_ylist)
	print train_loss_weight[-3:]
	print val_loss_weight[-3:]
	print ("processing...")

	for epoch in range(epoch_num):
		startime = time.time()
		#train
		model.train()
		train_loss = 0
		data_train = Dataset(train_xlist,train_ylist)
		data_mix = mix_Dataset(mtrain_xlist)
		train_loader = DataLoader(dataset = data_train, batch_size = 5, shuffle = True)
		mix_loader = DataLoader(dataset = data_mix, batch_size = 5, shuffle = True)

		for step, ((batch_x, batch_y), batch_m) in enumerate(zip(train_loader, mix_loader)):
			x = Variable((batch_x + batch_m).cuda(), requires_grad = True)
			y = Variable(batch_y.cuda(), requires_grad = True)

			y_pred, bm = model(x)
			"""loss_weight = torch.ones(y_pred.size(),dtype = torch.double)
			loss_weight[:,:,-1] = 1"""
			loss_fn = nn.BCEWithLogitsLoss(pos_weight = train_loss_weight.cuda())
			loss = loss_fn(y_pred,y)

			opt.zero_grad()
			loss.backward()
			opt.step()
			train_loss += loss.item()
		train_loss /= step

		#val

		val_loss = 0
		model.eval()
		data_val = Dataset(val_xlist,val_ylist)
		data_mix = mix_Dataset(mval_xlist)
		val_loader = DataLoader(dataset = data_val, batch_size = 5, shuffle = False)
		mix_loader = DataLoader(dataset = data_mix, batch_size = 5, shuffle = True)

		for step, ((batch_x, batch_y), batch_m) in enumerate(zip(val_loader,mix_loader)):
			x = Variable((batch_x + batch_m).cuda(), requires_grad = False)
			y = Variable(batch_y.cuda(), requires_grad = False)
			y_pred, bm = model(x)

			"""loss_weight = torch.ones(y_pred.size(), dtype = torch.double)
			loss_weight[:,:,-1] = 1"""
			loss_weight = pos_weight(val_ylist)
			loss_fn = nn.BCEWithLogitsLoss(pos_weight = val_loss_weight.cuda())

			loss = loss_fn(y_pred,y)
			val_loss += loss.item()

		if epoch == 0:
			np.save("0.npy",y_pred.data.cpu().numpy())
		if epoch > 0:
			np.save("1.npy",y_pred.data.cpu().numpy())
			np.save("ans.npy", y.data.cpu().numpy())

		val_loss /= step


		print("epoch: ", epoch ," | train_loss: " ,"%.4f" % train_loss, " | val_loss: " ,"%.4f" % val_loss, " | time = " ,int(time.time()-startime) )

		if val_loss < min_loss:
			min_loss = val_loss
			min_epoch = epoch
			print("min epoch: " ,min_epoch)
		torch.save(model.state_dict(), "cnnparams.pkl")

def main(cuda_num):

	train_xlist = []
	train_ylist = []
	val_xlist = []
	val_ylist = []
	mtrain_xlist = []
	mval_xlist = []

	data_dir = "/home/lego/guitar_solo_detection/feature/guitarset/"

	for n in range (1,181):
		x_data = np.load(data_dir+"x"+str(n)+".npy")
		y_data = np.load(data_dir+"y"+str(n)+".npy")
		m_data = np.load(data_dir+"m"+str(n)+".npy")
		if n <= 120:
			train_xlist.append(np.array(x_data, dtype=float)[np.newaxis,:])
			mtrain_xlist.append(np.array(m_data, dtype=float)[np.newaxis,:]*0.4)
			train_ylist.append(np.array(y_data, dtype=float)[:])
		elif n <= 160:
			val_xlist.append(np.array(x_data, dtype=float)[np.newaxis,:])
			mval_xlist.append(np.array(m_data, dtype=float)[np.newaxis,:]*0.4)
			val_ylist.append(np.array(y_data, dtype=float)[:])

	with torch.cuda.device(cuda_num):
		train(train_xlist, train_ylist, val_xlist, val_ylist, mtrain_xlist, mval_xlist)

cuda_num = 0
main(cuda_num)





