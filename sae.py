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
trainloader, validloader = dataloader(colab=False, batch_size=2)

net = TimeNet()
net.double()
# net.load_state_dict(torch.load('weights/model-sae3-checkpoint.pt'))
if CUDA: 
    net.cuda()

criterion = nn.MSELoss()
learning_rate = 0.006
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

best_train_loss = 1000
best_val_loss = 1000
lossArr = []
for i in range(EPOCHS):
    train_loss_value = 0
    val_loss_value = 0
    train_loss_epoch = np.array([])
    val_loss_epoch = np.array([])

    # training
    for inputs in tqdm(trainloader):
        input_ = Variable(torch.DoubleTensor(inputs), requires_grad=False)
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
        
        train_loss_value = loss.data.cpu().numpy() * input_.size(0) if CUDA else loss.data.numpy() * input_.size(0)
        # lossArr = np.append(lossArr, [train_loss_value], axis=0)
        train_loss_epoch = np.append(train_loss_epoch, [train_loss_value], axis=0)

        loss.backward()
        optimizer.step()
    print("epoch: %s, loss: %s" % (i, train_loss_value))
    train_loss_epoch = np.average(train_loss_epoch)
    print('current_loss: %s, best_train_loss: %s' % (train_loss_epoch, best_train_loss))
    
    # validation
    for inputs in tqdm(validloader):
        input_ = Variable(torch.DoubleTensor(inputs), requires_grad=False)
        if CUDA: 
            input_ = input_.cuda()
        input_reversed = input_.data.cpu().numpy() if CUDA else input_.data.numpy()
        input_reversed = np.flip(input_reversed, axis=1).copy()
        input_reversed = Variable(torch.from_numpy(input_reversed).double(), requires_grad=False)
        if CUDA: 
            input_reversed = input_reversed.cuda()

        predicted, encoded = net(input_, input_reversed)
        loss = criterion(predicted, input_reversed)

        val_loss_value = loss.data.cpu().numpy() * input_.size(0) if CUDA else loss.data.numpy() * input_.size(0)
        val_loss_epoch = np.append(val_loss_epoch, [val_loss_value], axis=0)
    print("epoch: %s, loss: %s" % (i, val_loss_value))
    val_loss_epoch= np.average(val_loss_epoch)
    print('current_loss: %s, best_val_loss: %s' % (val_loss_epoch, best_val_loss))


    if (val_loss_epoch < best_val_loss):
        best_val_loss = val_loss_epoch
        if CUDA:
            torch.save(net.cpu().state_dict(), 'weights/timenet.pt')
            net.cuda()
        else:
            torch.save(net.state_dict(), 'weights/timenet.pt')
    # torch.save(net.state_dict(), 'weights/model-sae3-checkpoint.pt')
    # np.savetxt('loss.csv', lossArr, delimiter=',')

# if CUDA:
#     net.cpu()
# torch.save(net.state_dict(), 'weights/model-sae3.pt')


# if __name__ == '__main__':
    # print('lol')