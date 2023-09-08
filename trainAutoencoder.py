import torch
from modelLib import *
from environment import *
from dataLoader import *
import os
from matplotlib import pyplot as plt
from pytorch_model_summary import summary
from PIL import Image

if not os.path.exists(RESULTSDIR):
    os.makedirs(RESULTSDIR)

# Initialize data loader
dataset_train = myDataset_npy_list(path_list=DATADIR_LIST, source='train',val_split=0.1)
# dataset_train = myDataset_npy(path=DATADIR, source='train',val_split=0.1)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCHSIZE,
        num_workers=0, pin_memory=True, drop_last=True)

dataset_val = myDataset_npy_list(path_list=DATADIR_LIST,source='validation',val_split=0.1)
# dataset_val = myDataset_npy(path=DATADIR,source='validation',val_split=0.1)
loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=BATCHSIZE,
        num_workers=0, pin_memory=True, drop_last=True)


# Initialize ML Model
print(DEVICE)
model = autoencoder_t1().to(DEVICE)
print(summary(model, torch.zeros(1,3,64,64).to(DEVICE)))

reconLoss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-2, weight_decay=1e-5)

lossTrain = []
lossVal = []
if EPOCH0>0:
    model, optimizer, _, lossTrain, lossVal = loadChkPt(f'chkpt_{EPOCH0-1}.pt', model, optimizer)

# Train
for epoch in range(NEPOCHS):
    print(f'Epoch: {EPOCH0+epoch}')
    lossTrain_s = 0
    lossVal_s = 0
    
    model.train()
    for i, imgs in enumerate(loader_train):
        imgs = imgs.to(DEVICE)
        optimizer.zero_grad()
        ls, reconImgs = model(imgs)
        if epoch==(NEPOCHS-1):
            for idx in range(0,reconImgs.shape[0]):

                X = Image.fromarray(np.uint8(reconImgs.cpu().detach().numpy()[idx].transpose([1,2,0])[:,:,::-1]*255))
                X.save(f'./img/result_{idx}.jpg')
                X = Image.fromarray(np.uint8(imgs.cpu().detach().numpy()[idx].transpose([1,2,0])[:,:,::-1]*255))
                X.save(f'./img/original_{idx}.jpg')
        loss = reconLoss(reconImgs, imgs)
        # print(f'train epoch {epoch} index {i} loss {loss}')
        lossTrain_s += loss.item()
        loss.backward()
        optimizer.step()  

    # Validation
    model.eval()
    with torch.no_grad():
        for i, imgs in enumerate(loader_val):
            imgs = imgs.to(DEVICE)
            _, reconImgs = model(imgs)
            loss = reconLoss(reconImgs, imgs)
            # print(f'train epoch {epoch} index {i} loss {loss}')
            lossVal_s+=loss.item()

    lossTrain.append(lossTrain_s/len(loader_train))
    lossVal.append(lossVal_s/len(loader_val))

    print(f'Train Loss: {lossTrain[-1]}, Val Loss: {lossVal[-1]}')

    if (EPOCH0+epoch+1)%SAVEINTERVAL==0:
        chkpt = {'epoch': EPOCH0+epoch,
                 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'lossTrain': lossTrain,
                 'lossVal': lossVal}
        saveChkPt(chkpt, f'chkpt_soft_{EPOCH0+epoch}.pt')

        plt.figure()
        plt.plot(range(EPOCH0+epoch+1), lossTrain, label='Train')
        plt.plot(range(EPOCH0+epoch+1), lossVal, label='Validation')
        plt.legend()
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.savefig(os.path.join(RESULTSDIR,"trainAutoencoder.png"))
        plt.close()
