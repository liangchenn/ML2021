import numpy as np
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TIMITDataset(Dataset):

    def __init__(self, X, y):
        self.data = torch.from_numpy(X).float()
        
        if y is not None:
            y = y.astype(int) 
            self.label = torch.LongTensor(y)
        else:
            self.label = None

        

    def __getitem__(self, index):

        if self.label is not None:
            # Train and Val dataset
            return self.data[index], self.label[index]
        else:
            # Test dataset
            return self.data[index]


    def __len__(self):
        return len(self.data)


class TIMITClassifier(nn.Module):
    def __init__(self, mode=1):
        super().__init__()


        self.mode = mode
        self.mpath = "./model{}.pth".format(mode)

        # model architecture
        self.net1 = nn.Sequential(
            nn.Linear(429, 1024),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 1024),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),

            nn.Linear(512, 128),
            nn.Sigmoid(),
            nn.Linear(128, 39)
        )

        self.net2 = nn.Sequential(
            nn.Linear(429, 1024),
            nn.BatchNorm1d(1024),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Sigmoid(),
            nn.Linear(128, 39)
        )

        self.net3 = nn.Sequential(
            nn.Linear(429, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Sigmoid(),
            nn.Linear(128, 39)
        )


        # loss function
        self.criterion = nn.CrossEntropyLoss()


    def cal_loss(self, pred, y, L2=False):
        '''
        Loss with L2-regularization.
        '''
        if L2:

            l2_lambda = 0.0001
            l2_reg = 0

            for param in model.parameters():
                l2_reg += torch.sum(param.pow(2))

            loss = self.criterion(pred, y) + l2_lambda * l2_reg

        else:
            loss = self.criterion(pred, y)

        return loss


    def forward(self, x):

        if self.mode == 1:
            return self.net1(x)
        elif self.mode == 2:
            return self.net2(x)
        else:
            return self.net3(x)



def train_val_split(X, y, shuffle=False, ratio=0.2):
    '''
    train/validation split with val ratio provided and data shuffled.
    '''

    if shuffle:
        idx = np.random.permutation(X.shape[0])
        X, y = X[idx], y[idx]

    percent = int(X.shape[0]*(1-ratio))

    return X[:percent], y[:percent], X[percent:], y[percent:]


#check device
def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# train model
def train(model, train_loader, val_loader, num_epoch, lr, device, L2=False):

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(num_epoch):

        train_acc = 0.0
        val_acc   = 0.0
        train_loss = 0.0
        val_loss   = 0.0
        

        # training part
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optim.zero_grad()                       # remove previous grad
            outputs = model(x)                      # prediction prob. in (1, 39) shape
            loss = model.cal_loss(outputs, y, L2)       # calculate loss
            loss.backward()                         # back propagation
            optim.step()                            # update params

            _, pred = torch.max(outputs, 1)         # get category with max prob., only need the index

            # update total loss
            train_acc += (pred.cpu() == y.cpu()).sum().item()
            train_loss += loss.item()

        
        # validation
        model.eval()
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = model.cal_loss(outputs, y, L2)
                _, pred = torch.max(outputs, dim=1)

                val_acc += (pred.cpu() == y.cpu()).sum().item()
                val_loss += loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
            ))

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model.mpath)
                print("saving model with acc: {:.3f}".format(best_acc/len(val_set)))

    return model


def predict(model, test_loader, device):

    predicts = []
    model.eval()
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            outputs = model(x)
            _, pred = torch.max(outputs, 1)

            for y in pred.cpu().numpy():
                predicts.append(y)

    return predicts


def save_pred(pred:list, filepath:str):

    with open(filepath, 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(pred):
            f.write('{},{}\n'.format(i, y))

    print('Finish saving prediction at {}'.format(filepath))


def post_process(pred:list, window=2):

    posts = []
    new_pred = pred[:]

    for i, p in enumerate(new_pred):
    
        if i <= window or i + window >= (len(new_pred) - 1):
            continue
        
        
        tmp = new_pred[i-window:i] + new_pred[i+1:i+(window+1)]

        if len(set(tmp)) == 1 and p not in set(tmp):
            posts.append({"i" : i, "value" : p})
            new_pred[i] = tmp[0]


    return new_pred
