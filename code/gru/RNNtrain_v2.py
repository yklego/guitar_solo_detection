import torch 
import torch.nn as nn
import os
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

hidden_size = 3
input_size = 21
output_size = 1



class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, dropout = 0.2)
        self.linear = nn.Linear(hidden_size, output_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):

        input = torch.transpose(input, 0, 1).cuda()
        input = input.view(-1,1,21)

        output_m, h_n = self.gru(input, hidden.cuda())
        # output of shape (seq_len, batch, num_directions * hidden_size)
        reshape = output_m.view(input.size()[0],1,1,hidden_size)
        reshape = reshape[:,0,0,:]

        output = self.linear(reshape)
        output = self.sigmoid(output)

        return output

    def initHidden(self, seq_len):
        return torch.zeros(1, 1, self.hidden_size, dtype=torch.float)


learning_rate = 0.001
loss_fn = nn.MSELoss()

def train(input_tensor, target, model):

    #input shape : (seq_len, batch, input_size)

    model.train()
    model.zero_grad()

    h0 = model.initHidden(input_tensor.size()[1])
    output = model(input_tensor, h0)

    output = output[:,0]
    loss = loss_fn(output.cuda(), target.cuda())
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in model.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() #?

def val_loss_estimate(input_tensor, target, model):

    model.eval()
    h0 = model.initHidden(input_tensor.size()[1])
    output = model(input_tensor,h0)

    output = output[:,0]

    loss = loss_fn(output.cuda(), target.cuda())

    return loss.item()


def main():

    n_epoch = 30
    min_val_loss = 100
    best_epoch = 0
    matplot=[],[]

    for n in range (n_epoch): 
        val_loss = 0
        model = GRU(input_size, hidden_size, output_size)
        model.cuda()
        for root, dirr, file in os.walk('feature/basic/train/'):
            shuffle(file)
            for filename in file:
                if 'x' in filename:
                    num = filename.strip('x.npy')
                    filepath = os.path.join(root,filename)
                    
                    print('Loading training data: #'+num+' ...')

                    input_tensor = torch.tensor( torch.from_numpy( np.load(filepath)).cuda(),dtype = torch.float)
                    target = torch.tensor( torch.from_numpy( np.load(filepath.replace("x", "y"))).cuda() ,dtype = torch.float)

                    train(input_tensor, target, model)

        for root, dirr, file in os.walk('feature/basic/val/'):
            shuffle(file)
            for filename in file:
                if 'x' in filename:
                    num = filename.strip('x.npy')
                    filepath = os.path.join(root,filename)
                    
                    print('Testing val data: #'+num+' ...')

                    input_tensor = torch.tensor( torch.from_numpy(np.load(filepath)).cuda() ,dtype = torch.float)
                    target =torch.tensor( torch.from_numpy(np.load(filepath.replace("x", "y"))).cuda() ,dtype = torch.float)

                    val_loss += val_loss_estimate(input_tensor, target, model)
        print("epoch #%d" % (n))
        print("val loss = %.4f" % (val_loss))
        print("min val loss = %.3f" % (min_val_loss) )
        print("best epoch = %d" % (best_epoch))
        matplot[0].append(n)
        matplot[1].append(val_loss)
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_epoch = n
            torch.save(model.state_dict(), 'params.pkl')
            print("******************************")
            print("change best epoch to %d" % (n) )
            print("change min val loss = %.3f" % (min_val_loss) )
    f = open("training_detail/1.txt", "w")
    f.write(matplot[0])
    f.write(matplot[1])
    """
    plt.plot(matplot[0],matplot[1])
    plt.xlabel("epoch")
    plt.ylabel("val loss")
    plt.text(2, 0.65, "best epoch = %d" % (best_epoch))
    plt.show()
    """

with torch.cuda.device(1):
    main()

