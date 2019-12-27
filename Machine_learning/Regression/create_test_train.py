
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler,Normalizer, PowerTransformer
from sklearn.compose import ColumnTransformer
from pathlib import Path
import pandas as pd
import seaborn as sns


import sys, os
from datetime import date

# Create a random dataset
rng = np.random.RandomState(1)


class Plotter:
    def __init__(self, data, Run_name, arg):

        self.data = data
        self.Run_name = Run_name
        self.arg = arg

        # instance variables

    def plot(self,colvar): #colvar is x or y
        all_plots="junta_plots/violin_plots/"+self.arg+"/"+self.Run_name
        if not os.path.exists(all_plots):
            os.makedirs(all_plots)

        sns.violinplot(x=colvar+"_val", y="entries", data=self.data)
        vplotname = all_plots+"/"+self.arg+"_violin_"+str(date.today())+"_"+colvar+"_val_"+self.Run_name+".png"
        plt.xlabel(r"$\sigma_"+colvar+"$",fontsize="x-large")
        plt.suptitle(self.arg)
        plt.savefig(vplotname)
        print(vplotname+" is created")

        plt.close()

    def create_pairplots(self):
        df = self.data.copy()
        sns.pairplot(df)
        plotdir = "all_plots/pair_plots"

        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
        plotname = plotdir+"/"+"pairplot_"+str(date.today())+"_"+self.Run_name+".png"
        plt.savefig(plotname)
        print(plotname+" is created")
        plt.close()




class CreateCombineWithMean:
    def __init__(self, data,  Run_name,arg):
        self.data = data
        self.Run_name = Run_name
        self.arg = arg


    def plot_means(self,sigy, y_entries):
        all_plots = "junta_plots/mean_plots/"+self.arg+"/"+self.Run_name
        if not os.path.exists(all_plots):
            os.makedirs(all_plots)

        plt.scatter(sigy,y_entries )
        plt.suptitle(self.arg+" dataset")
        plt.xlabel(r"$\sigma_y$", fontsize="x-large")
        plt.ylabel(r"$mean_{entries}$",fontsize="x-large")
        imname = all_plots+"/"+"check_yentries_mean_"+self.arg+"_"+str(date.today())+"_"+self.Run_name+".png"
        plt.savefig(imname)
        print(imname+" is created")




    def create_hdf(self): #colvar has to be x or y
        data = self.data.copy()
        data_temp = data[["entries","y_val"]]
        data_temp = data_temp.sort_values(by=["y_val"])
        data_temp = data_temp.groupby(["y_val"], sort=False)["entries"].mean().reset_index()
        data["y_entries"]= 0
        data = data.rename(columns={"entries":"x_entries"})
        data = data[["x_entries","y_entries","x_val","y_val"]]

        for i,sigy in enumerate(data.iloc[:,3]):
            for j,sigy_temp in enumerate(data_temp.iloc[:,0]):
                if sigy == sigy_temp:
                    data.iloc[i,1] = data_temp.iloc[j,1]

        sigy = data["y_val"].to_numpy()
        y_entries = data["y_entries"].to_numpy()
        self.plot_means(sigy, y_entries)

        all_hdfs = "junta_hdfs/"+self.arg+"/"+Run_name
        if not os.path.exists(all_hdfs):
            os.makedirs(all_hdfs)
        hdfname= all_hdfs+"/"+"combined_"+self.arg+"_"+str(date.today())+"_"+Run_name+".h5"
        data.to_hdf(hdfname,key="df")
        print(hdfname+" is created")





def create_train_test_split_with_mean(infile, Run_name): #infile pathlib format

    data = pd.read_hdf(str(fpath), key="df")
    X_train,X_test=train_test_split(data ,test_size=0.2)

    # initiate instances of Plotter class
    plotter_train = Plotter(data, Run_name, arg="Train")
    plotter_train.plot(colvar="x")
    plotter_train.plot(colvar="y")

    plotter_test = Plotter(data, Run_name, arg="Test")
    plotter_test.plot(colvar="x")
    plotter_test.plot(colvar="y")

    print("plotter ", type(plotter_train.plot(colvar="y")))

    hdf_train = CreateCombineWithMean(X_train,Run_name,arg="Train")
    hdf_train.create_hdf()

    hdf_test = CreateCombineWithMean(X_test,Run_name,arg="Test")
    hdf_train.create_hdf()




#Run_name="Nov21"
Run_name="Nov20"
#Run_name="Nov20_Nov21"
p = Path.home()
fname="combined_"+Run_name+".h5"   #previously fname="combined_new.h5"

#when nov20_21
#fname =Run_name+".h5"

if __name__ == '__main__':
    fpath = p/"numpy_cnn/hdf_files"/fname
    create_train_test_split_with_mean(fpath, Run_name=Run_name)



