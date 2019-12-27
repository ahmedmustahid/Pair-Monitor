from __future__ import print_function, division
import os
import time
import torch
import pandas as pd
#from skarr import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchsummary import summary

from pathlib import Path
from datetime import date
from datetime import datetime
import sys,re

from sklearn.metrics import confusion_matrix

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

writer = SummaryWriter()

class mydataset(Dataset):

    def __init__(self, hdf_filepath,transform=None):

        self.df = pd.read_hdf(str(hdf_filepath), key="df")
        self.transform = transform

    def __len__(self):
        return self.df.shape[1]


    def __getitem__(self, col_id):
        if torch.is_tensor(col_id):
            col_id = col_id.tolist()


        arrpaths = self.df.iloc[:,col_id]

        labels=[]
        arrs=[]
        for i,arrpath in enumerate(arrpaths):
            arr = np.load(arrpath)
            if self.transform:
                arr = self.transform(arr)

            label = col_id
            labels.append(label)
            arrs.append(arr)


        return (arrs, labels)

class ToTensor(object):

    def __call__(self, array):

        array = np.expand_dims(array,axis=0)
        return torch.from_numpy(array).float()


class CreateDataLoader:

    def __init__(self,filepath,str_type="Train", batch_len=5):

        self.filepath = filepath
        self.str_type = str_type
        self.batch_len = batch_len


    def create_dataloader(self):
        #transformed_ds = mydataset(hdf_filepath=str(filepath), transform = transforms.Compose([transforms.Grayscale(num_output_channels=1) , transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,) ) ]))
        transformed_ds = mydataset(hdf_filepath=str(self.filepath), transform=transforms.Compose([ToTensor()]))

        arrs=[]
        labels=[]

        data=[]
        for column_id in range(len(transformed_ds)):
            im, lab = transformed_ds[column_id]
            for i in range(len(im)):
                data.append([im[i], lab[i]])

            arrs.append(im)
            labels.append(lab)

        labels = np.array(labels)

        if self.str_type=="Train":
            print(self.str_type +" data len ",len(data))
            dataloader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=self.batch_len)

        elif self.str_type=="Test":
            print("testdata len ",len(data))
            dataloader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=self.batch_len)


        i1, l1 = next(iter(dataloader))
        print(i1.shape)

        return dataloader






class VGG16(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super(VGG16, self).__init__()

        self.block_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),stride=(1,1),padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),stride=(1,1),padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),stride=(1,1),padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
                )

        #self.block_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1),
        #        nn.ReLU(),
        #        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),stride=(1,1),padding=1),
        #        nn.ReLU(),
        #        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        #        )


        #self.block_3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1),
        #        nn.ReLU(),
        #        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3),stride=(1,1),padding=1),
        #        nn.ReLU(),
        #        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        #        )

        #self.block_4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1),
        #        nn.ReLU(),
        #        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3),stride=(1,1),padding=1),
        #        nn.ReLU(),
        #        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        #        )


        #self.block_5 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1),
        #        nn.ReLU(),
        #        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3),stride=(1,1),padding=1),
        #        nn.ReLU(),
        #        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        #        )


        self.classifier = nn.Sequential(
                nn.Linear(64*32*32, 1024),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(1024,512),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(512,num_classes)
                )

        #pytorch initializes automatically
        #https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
        #for m in self.modules():
        #    if isinstance(m , torch.nn.Conv2d):
        #        m.weight.detach().normal_(0,0.05)

    def forward(self,x):

        x = self.block_1(x)
        #x = self.block_2(x)

        #x = self.block_3(x)
        #x = self.block_4(x)
        #x = self.block_5(x)

        #print("x shape ", x.shape)
        logits = self.classifier(x.view(-1,64*32*32))
        probs = F.softmax(logits, dim=1)

        return logits, probs

def compute_accuracy(model, data_loader):
    correct_pred, num_examples, data_loss = 0, 0, 0
    for i, (features, targets) in enumerate(data_loader):

        features = features.to(device)
        targets = targets.to(device)

        logits, probs = model(features)

        cost = F.cross_entropy(logits, targets)

        data_loss += cost.item()

        _, predicted_labels = torch.max(probs, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100, data_loss



def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    imgtemp = img.cpu()
    npimg = imgtemp.numpy()

    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def arrs_to_probs(net, arrs):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of arrs
    '''
    #net = net.to(dtype=torch.float, device = torch.device("cpu"))
    net = net.cuda()
    output, probs = net(arrs.to(device))


    _, preds_tensor = torch.max(output, 1)
    preds_tensortemp = preds_tensor.cpu()
    preds = np.squeeze(preds_tensortemp.numpy())

    return torch.from_numpy(preds), [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_confusion(confusion_mat,fname="conf_mat"):
    fig,ax = plt.subplots(figsize=(11,8))
    ax.set(xticks= np.arange(confusion_mat.shape[1]), yticks= np.arange(confusion_mat.shape[0]+1), xticklabels = classes, yticklabels= classes, title=None, ylabel="True label", xlabel="Predicted label")
    im = ax.imshow(confusion_mat, interpolation="nearest", cmap= plt.cm.Blues)
    ax.figure.colorbar(im,ax=ax)


    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = 'd'
    thresh = confusion_mat.max() / 2.
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            ax.text(j, i, format(confusion_mat[i, j], fmt),ha="center", va="center",color="white" if confusion_mat[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(fname+".png")
    plt.close()


def plot_classes_preds(model, arrs, labels):
    preds, probs = arrs_to_probs(model, arrs)
    # plot the arrs in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(48,12))
    arrs = arrs.cpu().numpy()
    arrs = arrs.reshape(-1,64,64)
    for idx in np.arange(9):
        ax = fig.add_subplot(3, 3, idx+1, xticks=[], yticks=[])
        ax.imshow(arrs[idx], interpolation="bicubic", cmap="nipy_spectral")
        #matplotlib_imshow(arrs[idx])
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    plt.savefig("pred_vs_true_4classes_rasca_batch"+str(num_batch)+"_epoch"+str(num_epochs)+"_classes"+str(num_classes)+"_mydata"+".png")
    return fig


def check_labels(dataloader):
    print("Size ", len(dataloader))
    input1, label1 = next(iter(dataloader))
    print("label ", label1)
    print("label size ", label1.size())


class Plotter:

    def __init__(self, epochs, train_nums, val_nums,x_label,y_label,str_type="Losses"): #str_type=>"Lossses" or "accuracy"

        self.epochs = epochs
        self.train_nums = train_nums #train accs or losses
        self.val_nums = val_nums
        self.x_label = x_label
        self.y_label = y_label
        self.str_type = str_type

    def plotter(self):

        plt.plot(self.epochs, self.train_nums, label="Train")
        plt.plot(self.epochs, self.val_nums, label="Test")
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

        plt.xticks(ticks=epochs)
        plt.legend()
        plotdir="cnn_plots_"+str(date.today())

        if not os.path.exists(plotdir):
            os.mkdir(plotdir)
        plotname= plotdir+"/"+self.str_type+"_"+str(num_batch)+"_epoch"+str(num_epochs)+"_"+str(date.today())+".png"
        plt.savefig(plotname)
        print(plotname+" is created")
        plt.close()


def compute_pred_per_batch(model, validloader):
    predlist = torch.zeros(0, dtype= torch.long, device="cpu")
    lablist = torch.zeros(0, dtype= torch.long, device="cpu")
    for batch_idx, (features, targets) in enumerate(validloader):
        preds, _ = arrs_to_probs(model, features)
        predlist = torch.cat([predlist, preds.view(-1).cpu()])
        lablist = torch.cat([lablist, targets.view(-1).cpu()])

    return (predlist, lablist)








if __name__=="__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    # Hyperparameters
    learning_rate = 0.001
    num_epochs = 30

    # GLOBAL VARIABLES
    num_features = 64*64
    #num_classes = 16
    num_classes = 25
    num_batch=32


    model = VGG16(num_features=num_features, num_classes= num_classes)
    model.to(device)
    summary(model, (1,64,64))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    start_time = time.time()
    p = Path.home()

    test_filepath =p/"numpy_cnn/hdfs_2019-12-26"/"hdfs_2019-12-26_all_test.h5"
    train_filepath =p/"numpy_cnn/hdfs_2019-12-26"/"hdfs_2019-12-26_all_train.h5"


    TrainLd = CreateDataLoader(train_filepath,"Train",num_batch)
    trainloader = TrainLd.create_dataloader()

    TestLd = CreateDataLoader(test_filepath,"Test",num_batch)
    validloader = TestLd.create_dataloader()

    torch.save(trainloader, "trainloader"+"_"+str(datetime.now().time()))
    torch.save(validloader, "validloader"+"_"+str(datetime.now().time()))
    #sys.exit()

    print("check train labels:")
    check_labels(trainloader)
    print("_______________________________________")
    print("check test labels:")
    check_labels(validloader)

    epochs = []
    train_accs=[]
    val_accs=[]

    train_losses=[]
    val_losses=[]
    running_loss=0.0


    df = pd.read_hdf(test_filepath, key="df")
    print(type(df.columns))
    print(df.columns[1])

    classes=[]
    for col in df.columns:

        sigx = col.split('_')[0]
        sigy = col.split('_')[1]

        x_val = next(iter(re.findall("\d+\.\d+",sigx)))
        y_val = next(iter(re.findall("\d+\.\d+",sigy)))
        val = "x"+str(x_val)+"_"+"y"+str(y_val)

        classes.append(val)
    print("classes", classes)

    #FOR Confusion matrix
    predlist = torch.zeros(0, dtype= torch.long, device="cpu")
    lablist = torch.zeros(0, dtype= torch.long, device="cpu")

    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(trainloader):

            features = features.to(device)
            targets = targets.to(device)

            ### FORWARD AND BACK PROP
            logits, probas = model(features)
            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()

            cost.backward()

            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            running_loss += cost.item()
            ### LOGGING
            if not batch_idx % 50:
                print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                       %(epoch+1, num_epochs, batch_idx,
                         len(trainloader), cost))

               # ...log the running loss
                writer.add_scalar('training loss',
                                running_loss / 100,
                                epoch * len(trainloader) + batch_idx)

                # ...log a Matplotlib Figure showing the model's predictions on a
                # random mini-batch

                #writer.add_figure('predictions vs. actuals',
                #                plot_classes_preds(model.cpu(), features, targets),
                #                global_step=epoch * len(trainloader) + batch_idx)

                running_loss = 0.0

            with torch.no_grad():
               if epoch==num_epochs-1:

                    #preds, _ = arrs_to_probs(model, features)
                    #print("type pred ", type(preds))
                    #print("preds shape ", preds.shape)
                    #predlist = torch.cat([predlist, preds.view(-1).cpu()])
                    #lablist = torch.cat([lablist, targets.view(-1).cpu()])

                    predlist, lablist = compute_pred_per_batch(model, validloader)
                    if batch_idx== len(trainloader)-1:
                        #plot_classes_preds(model.cpu(), features, targets)
                        print("inside plot_classes")


        model.eval()
        #with torch.set_grad_enabled(False): # save memory during inference
        with torch.no_grad(): # save memory during inference

            epochs.append(epoch)

            train_acc, train_loss = compute_accuracy(model, trainloader)
            val_acc, val_loss = compute_accuracy(model, validloader)

            train_accs.append(train_acc)
            val_accs.append(val_acc)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print('Epoch: %03d/%03d | Train: %.3f%% | Valid: %.3f%%' % (
                  epoch+1, num_epochs,
                  train_acc,
                  val_acc))

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    #WRITE ARCHITECTURE
    arrs, labels = next(iter(trainloader))
    #img_grid = torchvision.utils.make_grid(arrs)
    model = model.to(dtype=torch.float, device = torch.device("cpu"))
    writer.add_graph(model, arrs)


    #SAVE MODEL
    torch.save(model.state_dict(), "model"+"_"+str(datetime.now().time()))

    #MAKE PLOTS
    LossPlot= Plotter(epochs,train_losses, val_losses,"Epochs","Loss","Losses")
    LossPlot.plotter()

    AccPlot= Plotter(epochs,train_accs, val_accs,"Epochs","Accuracy","Accuracys")
    AccPlot.plotter()



    testpreds, testlabs = compute_pred_per_batch(model, validloader)

    conf_mat_train = confusion_matrix(lablist.numpy(), predlist.numpy())
    conf_mat_test = confusion_matrix(testlabs.numpy(), testpreds.numpy())

    plot_confusion(conf_mat_train, "conf_train_test5")
    plot_confusion(conf_mat_test, "conf_test_test5")

