from pathlib import Path
import numpy as np
#np.random.seed(0)
import pandas as pd
import torch
#torch.manual_seed(0)
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sys, os
from sklearn.utils import shuffle

from datetime import date

def create_hist(y_test, predictions):
    sigvals = [0.8, 1.0, 1.2, 1.4, 1.6]

    xy_colnums = [0,1]
    plotdir = Path.cwd()/"all_plots"/"nn_regplots"
    for colnum in xy_colnums:
        ax = plt.subplot()
        for j,sig in enumerate(sigvals):
            preds =[]
            for i in range(y_test.shape[0]):
               if abs(y_test[i,colnum]-sig)<0.1:
                    signame = str(sig)
                    preds.append(predictions[i,colnum])
            ax.hist(preds, bins=100, range=(0.6,2),alpha=0.5, label=signame, log=True, density = True)
            ticks = np.arange(0,2,step=0.2)
            ax.set_xticks(ticks)
            ax.set_xlim(0.5,2.0)

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), framealpha=0.3)


        if not os.path.exists(plotdir):
            os.mkdir(plotdir)

        if colnum==0:
            ax.set_xlabel(r"$\sigma_x$", fontsize="x-large")
        else:
            ax.set_xlabel(r"$\sigma_y$", fontsize="x-large")

        plotname = str(plotdir)+"/"+"nn_hist_"+str(colnum)+"_"+str(date.today())+".png"
        plt.savefig(plotname)
        print(plotname+" is created")
        plt.close()


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()

        #self.fc1 = nn.Linear(2, 144)
        #self.fc2 = nn.Linear(144, 72)
        #self.fc3 = nn.Linear(72, 18)
        #self.fc4 = nn.Linear(18, 2)

#good values for sigx
        self.fc1 = nn.Linear(2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 18)
        self.fc6 = nn.Linear(18, 2)
        #self.fc6 = nn.Linear(18, 1)


#following for sigy
        #self.fc1 = nn.Linear(4, 1024)
        #self.fc2 = nn.Linear(1024, 512)
        #self.fc3 = nn.Linear(512, 256)
        #self.fc4 = nn.Linear(256, 128)
        #self.fc5 = nn.Linear(128, 64)
        #self.fc6 = nn.Linear(64, 32)
        ##self.fc4 = nn.Linear(18, 2)
        #self.fc7 = nn.Linear(32, 1)



    def forward(self, x):


        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = self.fc4(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)

        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        #x = F.relu(self.fc5(x))
        #x = F.relu(self.fc6(x))
        #x = self.fc7(x)



        return x

def create_data_label(data_path):
    #p = Path.cwd()
    #fpath = p/"combined_new.h5"


    data = pd.read_hdf(str(data_path), key="df")
    data = shuffle(data)



    targets = data[["x_val", "y_val"]]
    features = data.drop(["x_val","y_val"], axis=1)

    scaler = MinMaxScaler()
    features = pd.DataFrame(scaler.fit_transform(features), columns= features.columns)


    return (features, targets)



p = Path.cwd()

train_path = p/"../hdf_files"/"combined_new_train.h5"
test_path = p/"../hdf_files"/"combined_new_test.h5"

X_test, y_test = create_data_label(test_path)
X_train, y_train = create_data_label(train_path)

#print("feature shape ", X_train.shape)
#print(X_test.shape)
#
#print("target shape ", y_train.shape)
#print(y_test.shape)


train_batch = np.array_split(X_train, 50)
label_batch = np.array_split(y_train, 50)

print("train batch len ", len(train_batch))
print("label batch len ", len(label_batch))

#print(train_batch[49])
#print(train_batch[49].to_numpy().shape)

print("label batch")
print(label_batch[49].to_numpy().shape)
print(label_batch[49])

for i in range(len(train_batch)):
    train_batch[i] = torch.from_numpy(train_batch[i].to_numpy()).float()
for i in range(len(label_batch)):
    label_batch[i] = torch.from_numpy(label_batch[i].to_numpy()).float()
    #label_batch[i] = torch.from_numpy(label_batch[i].to_numpy()).float().view(-1, 2)

#print("label_batch ", label_batch[49])
#print("label_batch shape ", label_batch[49].shape)


X_test = torch.from_numpy(X_test.to_numpy()).float()
print("X val ", X_test.shape)
y_test = torch.from_numpy(y_test.to_numpy()).float()
print("y val ", y_test.shape)
#y_test = torch.from_numpy(y_test.to_numpy()).float().view(-1, 2)



#print(len(train_batch))
#sys.exit()

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
model = Regressor()
#model.to(dtype= torch.float64, device = device)


#ps = model(train_batch[0])
#print(ps.shape)
#print(ps)
#sys.exit()
#model = Regressor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#epochs = 10
#epochs = 30
#epochs = 100
epochs =300

#device =

testlabels = []
preds = []

train_losses, test_losses = [], []
for e in range(epochs):
    model.train()
    train_loss = 0
    for i in range(len(train_batch)):
        optimizer.zero_grad()
        #model.to(device)
        output = model(train_batch[i])
        #output = model(train_batch[i].to(dtype= torch.float64, device= device))


        loss = criterion(output, label_batch[i])
        #loss = criterion(output, label_batch[i].to(dtype=torch.float64, device = device))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if e==epochs-1 and i==49:
            print(label_batch[i][:10])
            print("output prediction")
            print(output[:10])



    else:
        test_loss = 0
        accuracy = 0

        with torch.no_grad():
            model.eval()
            predictions = model(X_test)

            if e==epochs-1:
                create_hist(y_test.numpy(), predictions.numpy())

            #predictions = model(X_test.to(dtype= torch.float64, device= device))
            #if i==49:
            #    print("inside")
            #    print(predictions)
            #    print(predictions.shape)
            #test_loss += torch.sqrt(criterion(torch.log(predictions), torch.log(y_test)))

            test_loss += criterion(predictions, y_test)

        train_losses.append(train_loss/len(train_batch))
        test_losses.append(test_loss)



        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.7f}.. ".format(train_loss/len(train_batch)),
              "Test Loss: {:.7f}.. ".format(test_loss))


fig, ax = plt.subplots(figsize=(10,6))


plotdir = Path.cwd()/"all_plots"/"nn_regplots"
if not os.path.exists(plotdir):
    os.mkdir(plotdir)


epochlist= np.arange(1,epochs+1)
ax.plot(epochlist, train_losses, label="Train")
ax.plot(epochlist, test_losses, label="Test")
ax.set_xlabel("Epochs", fontsize="large")
ax.set_yscale("log")
#ax.set_xticks(np.arange(epochs))
#ax.set_yticks(np.arange(train_losses[0], test_losses[-1],step=0.000001))
ax.legend()
plt.suptitle("Mean Square Loss")
lossplot = str(plotdir)+"/"+"nn_reg_loss"+str(date.today())+".png"

plt.tight_layout()
plt.savefig(lossplot)
print(lossplot+" is created")
#plt.show()












