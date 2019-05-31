import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np 
import time
import os

class Network(nn.Module):
	def __init__ (self, input_size, hidden_size, output_size):
		super(Network, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size

		self.GRU = nn.GRU(input_size, hidden_size, 2, batch_first=True, dropout=0.5, bidirectional= True)
		self.linear = nn.Linear(hidden_size, output_size)

	def forward(self, x, h = None):  
		r, h = self.GRU(torch.transpose(x,1,2)) #(batch, seq, feature)
		r = r[:, :, :self.hidden_size] + r[:, :, self.hidden_size:]
		l = F.sigmoid(self.linear(r))
		output = l.view(l.size()[1])

		return output

	def init_hidden(self, bs): #?????
		weight = next(self.parameters()).data_nums
		return Variable(wwight.new(4, bs, 64).zero_())


def train(train_xlist, train_ylist, val_xlist, val_ylist, epoch_num=1000, lr=0.001):
	model = Network(84,4,1) 
	model.double()
	model.cuda()
	opt = optim.Adam(model.parameters(), lr=lr)

	minloss = 1000
	maxf1 = 0

	for epoch in range(epoch_num):
		start_time = time.time()

		#train
		train_loss = 0
		model.train()
		ind = np.arange(len(train_xlist))
		np.random.shuffle(ind)

		for i in ind:
			x = train_xlist[i][np.newaxis,:,:] #(batch, feature, seq)
			y = train_ylist[i]
			x = Variable(torch.from_numpy(x).cuda())
			y = Variable(torch.from_numpy(y).cuda())

			opt.zero_grad()
			y_pred = model(x)
			lossfn = F.binary_cross_entropy_with_logits(y_pred, y)
			lossfn.backward()
			opt.step()
			train_loss += lossfn.data[0]
		train_loss /= len(train_xlist)

		# validation part
		val_loss = 0
		model.eval()
		tp, fp, tn, fn = 0, 0, 0, 0
		for i in range(len(val_xlist)):
			x = val_xlist[i][np.newaxis,:,:]
			y = val_ylist[i]
			x = Variable(torch.from_numpy(x).cuda(), volatile=True)
			y = Variable(torch.from_numpy(y).cuda(), volatile=True)
			y_pred =model(x)

			lossfn = F.binary_cross_entropy_with_logits(y_pred, y)
			val_loss += lossfn.data[0]
		val_loss /= len(val_xlist)



		print('epoch:', epoch, '| train loss: %.4f' % train_loss, 'val_loss: %.4f' % val_loss, '| time:', int(time.time() - start_time), 'sec')

		if val_loss < minloss:
			minloss = val_loss
			minepoch = epoch
			print('min epoch:', minepoch, 'minloss: %4f' % minloss)
			torch.save(model.state_dict(), 'params.pkl')




def main(cuda_num):
	data_dir = '/home/bill317996/Guitar-Solo-Detection/feature/cqt/'
	target_dir = '/home/bill317996/Guitar-Solo-Detection/feature/basic/'
	train_xlist = []
	train_ylist = []
	val_xlist = []
	val_ylist = []
	test_xlist = []
	test_ylist = []

	"""
	for file in os.listdir(Train_dir):
		if "x" in file :
			xpath = os.path.join(Train_dir,file)
			ypath = xpath.replace('x','y')

			train_xlist.append(np.load(xpath))
			train_ylist.append(np.load(ypath))

	for file in os.listdir(val_dir):
		if "x" in file :
			xpath = os.path.join(val_dir,file)
			ypath = xpath.replace('x','y')

			val_xlist.append(np.load(xpath))
			val_ylist.append(np.load(ypath))

	for file in os.listdir(test_dir):
		if "x" in file :
			xpath = os.path.join(test_dir,file)
			ypath = xpath.replace('x','y')

			test_xlist.append(np.load(xpath))
			test_ylist.append(np.load(ypath))


	with torch.cuda.device(cuda_num):
		train(train_xlist, train_ylist, val_xlist, val_ylist)

cuda_num = 1
main(cuda_num)





	"""
	for i in range (60):

			data_num = i+1
			xname = "cqt" + str(data_num)+ ".npy"
			yname = "y" + str(data_num)+ ".npy"

			xpath = data_dir+xname
			ypath = target_dir+yname

			if data_num <= 40:
				train_xlist.append(np.array(np.load(xpath), dtype=float))
				train_ylist.append(np.load(ypath))

			elif data_num <= 50:
				val_xlist.append(np.array(np.load(xpath), dtype=float))
				val_ylist.append(np.load(ypath))

			#else:
			#	test_xlist.append(np.load(xpath))
			#	test_ylist.append(np.load(ypath))


	with torch.cuda.device(cuda_num):
		train(train_xlist, train_ylist, val_xlist, val_ylist)

cuda_num = 1
main(cuda_num)
