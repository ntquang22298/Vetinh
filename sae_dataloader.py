import os
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

datasets = [
    'bareland',
    'crop',
    'forest',
    'grass',
    'rice',
    'water'
]

class SAEDataloader(Dataset):
    def __init__(self, colab=False, root=''):
        self.root = root
        atensor = datamaker(self.root).clone().detach()
        self.data = torch.unsqueeze(atensor, 2)

    def __getitem__(self, index):
        ret = self.data[index]
        return ret

    def __len__(self):
        return self.data.size(0)


def datamaker(root):
    train_data = torch.tensor([], dtype=torch.double)
    for directory, subdir, files in os.walk(root):
        print(directory)
        # print(files)
        for file in files:
            filename = os.path.join(directory, file)
            print(file)
            with open(filename, 'r') as f:
                data = f.readlines()
            local_data = []
            for i in range(8, len(data)):
                element = data[i].strip().split()[1:]
                if 'NaN' in element: continue
                element = [float(e) for e in element]
                if len(element) == 0: continue
                local_data.append(element)
            local_data = np.array(local_data)
            local_data = normalize(local_data, norm='l2')
            local_data = torch.from_numpy(local_data)
            train_data = torch.cat((train_data, local_data), 0)

    return train_data


def dataloader(colab=False, batch_size=2):
    traindataset = SAEDataloader(colab=colab, root='sae_datasets/train')
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

    validdataset = SAEDataloader(colab=colab, root='sae_datasets/valid')
    validloader = DataLoader(validdataset, batch_size=batch_size, shuffle=True)
    return trainloader, validloader

def testloader(colab=False, batch_size=2):
    testdataset = SAEDataloader(colab=colab, root='sae_datasets/test')
    testloader = DataLoader(testdataset, batch_size=batch_size)

    return testloader

if __name__ == '__main__':
    # data = datamaker()
    # data= torch.tensor(data)
    # print(data[3])
    # data = torch.unsqueeze(data, 2)
    # print(data[3])
    # print(data.size())

    datamaker = SAEDataloader()
    print(len(datamaker))
    d = datamaker[0]
    print(d.size())