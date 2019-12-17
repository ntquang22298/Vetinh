# import pandas
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import random
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sae_dataloader import dataloader

from timenet import TimeNet

# PARAMS:
CUDA = torch.cuda.is_available()
EPOCHS = 100
BATCH_STEP_SIZE = 64

print('preparing data')
dataloader = dataloader(colab=False, batch_size=2)

net = TimeNet()
net.double()
# net.load_state_dict(torch.load('weights/model-sae3-checkpoint.pt'))
if CUDA: 
    net.cuda()

criterion = nn.MSELoss()
learning_rate = 0.006
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

lowest_loss_per_epoch = 1000
lossArr = []
for i in range(EPOCHS):
    total_iterations = 1
    loss_to_show = 0
    loss_per_epoch = np.array([])

    for inputs in tqdm(dataloader):
        input_ = Variable( torch.DoubleTensor(inputs), requires_grad=False )
        if CUDA: 
            input_ = input_.cuda()

        input_reversed = input_.data.cpu().numpy() if CUDA else input_.data.numpy()
        input_reversed = np.flip(input_reversed, axis=1).copy()
        input_reversed = Variable(torch.from_numpy(input_reversed).double(), requires_grad=False)
        if CUDA: 
            input_reversed = input_reversed.cuda()

        optimizer.zero_grad()
        predicted, encoded = net(input_, input_reversed)
        loss = criterion(predicted, input_reversed)
        
        loss_to_show = loss.data.cpu().numpy() * input_.size(0) if CUDA else loss.data.numpy() * input_.size(0)
        lossArr = np.append(lossArr, [loss_to_show], axis=0)
        loss_per_epoch = np.append(loss_per_epoch, [loss_to_show], axis=0)

        loss.backward()
        optimizer.step()
        total_iterations += 1
    # break
    print("epoch: %s, total_i: %s, loss: %s" % (i, total_iterations, loss_to_show))

    loss_per_epoch = np.average(loss_per_epoch)
    print('current_loss: %s, lowest_loss: %s' % (loss_per_epoch, lowest_loss_per_epoch))
    if (loss_per_epoch < lowest_loss_per_epoch):
        lowest_loss_per_epoch = loss_per_epoch
        if CUDA:
            torch.save(net.cpu().state_dict(), 'weights/timenet.pt')
            net.cuda()
        else:
            torch.save(net.state_dict(), 'weights/timenet.pt')
    # torch.save(net.state_dict(), 'weights/model-sae3-checkpoint.pt')
    np.savetxt('loss.csv', lossArr, delimiter=',')

# if CUDA:
#     net.cpu()
# torch.save(net.state_dict(), 'weights/model-sae3.pt')


# if __name__ == '__main__':
    # print('lol')