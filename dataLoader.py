import torch
from torchvision import transforms, utils, datasets
import os
from PIL import Image
from environment import *
import numpy as np

# A custom dataset to generate (Grayscale) image pairs
class myDataset(torch.utils.data.Dataset):
    def __init__(self, path=None, source=None, val_split=0):
        self.path = path

        allFiles = [0] if path==None else os.listdir(path)
        np.random.seed(RANDOMSEED)
        np.random.shuffle(allFiles)

        if DATALIMIT:
            allFiles = allFiles[:DATALIMIT]

        split = int(val_split*(len(allFiles)-1))

        # Filter irrelevant
        if source==None:
            print('Data:')
            self.files = allFiles
        elif source=="train":
            print('Training data:')
            self.files = allFiles[split:]
        elif source=="validation":
            print('Validation data:')
            self.files = allFiles[:split]

        print(len(self.files))
        self.transformIn = transforms.Compose([
                     transforms.Grayscale(num_output_channels=1),
                     transforms.Resize(64),
                     transforms.CenterCrop(64),
                     transforms.ToTensor()])
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.path,self.files[index]))
        inImg = self.transformIn(img)
        return inImg

    def loadFromFile(self, filepath):
        img = Image.open(filepath)
        inImg = self.transformIn(img)
        inImg = np.reshape(inImg, (1,1,64,64))
        return inImg


# A custom dataset to generate (Grayscale) image pairs
class myDataset_npy(torch.utils.data.Dataset):
    def __init__(self, path=None, source=None, val_split=0):
        self.path = path

        # allFiles = [0] if path == None else os.listdir(path)
        # np.random.seed(RANDOMSEED)
        # np.random.shuffle(allFiles)

        allFiles = np.load(path,allow_pickle=True)
        allFiles = allFiles.transpose(0,3, 1, 2)
        allFiles =allFiles.astype(np.float32)
        if DATALIMIT:
            allFiles = allFiles[:DATALIMIT]

        split = int(val_split * (len(allFiles) - 1))

        # Filter irrelevant
        if source == None:
            print('Data:')
            self.files = allFiles
        elif source == "train":
            print('Training data:')
            self.files = allFiles[split:]
        elif source == "validation":
            print('Validation data:')
            self.files = allFiles[:split]

        print(len(self.files))
        self.transformIn = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.files[index,:,:,:]



    # def loadFromFile(self, filepath):
    #     img = Image.open(filepath)
    #     inImg = self.transformIn(img)
    #     inImg = np.reshape(inImg, (1, 1, 64, 64))
    #     return inImg

# A custom dataset to generate (Grayscale) image pairs
class myDataset_npy_list(torch.utils.data.Dataset):
    def __init__(self, path_list=None, source=None, val_split=0):
        allFiles = None
        for path in path_list:
            if allFiles is None:
                allFiles = np.load(path,allow_pickle=True)
            else:
                temp = np.load(path,allow_pickle=True)
                allFiles = np.concatenate((allFiles,temp))
        print(allFiles.shape)
        np.random.seed(RANDOMSEED)
        np.random.shuffle(allFiles)
        allFiles = allFiles.transpose(0,3, 1, 2)
        allFiles =allFiles.astype(np.float32)
        if DATALIMIT:
            allFiles = allFiles[:DATALIMIT]

        split = int(val_split * (len(allFiles) - 1))

        # Filter irrelevant
        if source == None:
            print('Data:')
            self.files = allFiles
        elif source == "train":
            print('Training data:')
            self.files = allFiles[split:]
        elif source == "validation":
            print('Validation data:')
            self.files = allFiles[:split]

        print(len(self.files))
        self.transformIn = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.files[index,:,:,:]

# save checkpoint
def saveChkPt(state, filename):
    torch.save(state,os.path.join(RESULTSDIR,filename))
    return

# load checkpoint
def loadChkPt(filename, model, optimizer=None):
    chkpt = torch.load(filename)
    model.load_state_dict(chkpt['model'])
    if optimizer!=None: optimizer.load_state_dict(chkpt['optimizer'])
    loss_train = chkpt['lossTrain']
    loss_val = chkpt['lossVal']
    return model, optimizer, chkpt['epoch'], loss_train, loss_val
