import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score,precision_score, recall_score
import scalers

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

n_layers = 4
dropout_rate = 0.
patience = 2
torch.manual_seed(42)
class net(nn.Module):
    def __init__(self,  scaler=scalers.rank_gauss()):
        super(net, self).__init__()
        self.name = f'net scaler={scaler.name} n_layers={n_layers:d} lr={lr:.4f} dropout_rate={dropout_rate:.2f}'
        self.scaler = scaler
        modules = []
        for i in range(n_layers):
            modules.append(nn.Linear(200,200))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Linear(200, 1))
        self.fc = torch.nn.Sequential(*modules)
        print(self)

    def forward(self, x):
        x = self.fc(x)
        return x

    def fit(self, X, t, max_epochs=20):
        X_train, X_test, t_train, t_test = train_test_split(X, t, stratify=t, test_size=0.1, random_state=42)
        X_train = self.scaler.fit(X_train).transform(X_train)
        optimizer = optim.Adam(self.parameters())
        criterion = nn.BCEWithLogitsLoss()
        X_train = torch.from_numpy(X_train).float()
        t_train = torch.from_numpy(t_train).float()
        trainload = torch.utils.data.DataLoader(np.concatenate((t_train.unsqueeze(-1),X_train), axis = 1), batch_size=512,  shuffle=True)
        best_roc = 0.5
        best_epochs = 0
        init_state = self.state_dict()
        j = 0
        for i in range(max_epochs):
            for data in trainload:
                X = data[:,1:]
                t = data[:,0]
                optimizer.zero_grad()
                output = self(X)
                loss = criterion(output.squeeze(-1), t)
                loss.backward()
                optimizer.step()
            probas = self.predict_proba(X_test)[:,1]
            roc = roc_auc_score(t_test, probas)
            if (roc > best_roc):
                best_roc = roc
                best_epochs = i+1
                j = 0
            else:
                j = j+1
            print('Epoch',i,' roc:',roc)
            if j > patience:
                break
        trainload = torch.utils.data.DataLoader(np.concatenate((t.unsqueeze(-1),X), axis = 1), batch_size=512,  shuffle=True)
        self.load_state_dict(init_state)
        for i in range(best_epochs):
            for data in trainload:
                X = data[:,1:]
                t = data[:,0]
                optimizer.zero_grad()
                output = self(X)
                loss = criterion(output.squeeze(-1), t)
                loss.backward()
                optimizer.step()
            print('ReEpoch',i)
        return self
        
    def predict_proba(self, X):
        self.eval()
        X = self.scaler.transform(X)
        with torch.no_grad():
            X = torch.from_numpy(X).float()
            y = torch.sigmoid(self.forward(X)).cpu().numpy()
        return np.hstack((1-y,y))