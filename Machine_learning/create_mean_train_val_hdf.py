import numpy as np
from scipy.optimize import linprog
import pandas as pd
from pathlib import Path

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import cm
import sys, os, re
from matplotlib.colors import LogNorm
import seaborn as sns

from pathlib import Path

import ot

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def plotter(df):
    sns.violinplot(x="sigy",y="mean_entries",data=df)
    plt.xlabel(r"$\sigma_y$",fontsize="x-large")
    plt.ylabel(r"$Mean_{entries}$",fontsize="x-large")
    plt.suptitle(r"$\sigma_x$ constant: "+str(df.iloc[0,2]))

    plotdir = "plots"
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)
    plotname=plotdir+ "/all_entries_sigx_"+str(df.iloc[0,2])+"_.png"
    plt.savefig(plotname)
    print(plotname+" is created")
    plt.close()


def create_hdf(df):
    hdfdir="mean_entries_hdf"

    if not os.path.exists:
        os.mkdir(hdfdir)

    hdfname=hdfdir+"/all_entries_sigx_"+str(df.iloc[0,2])+"_.h5"

    df.to_hdf(hdfname,key="df")
    print(hdfname+" created")


def get_means(df):

    #print("df cols ",df.columns)
    #sys.exit()

    arr = df.to_numpy()
    #print("arr shape ", arr.shape)
    sections=5
    arrlist = np.array_split(arr, indices_or_sections=sections, axis=0)

    means =[]
    try:
        for arr in arrlist[:-1]:
                mean = np.mean(arr[:,0])
                means.append(mean)
                #print("arr shape inside for ", arr.shape)
                #sys.exit()


        df_temp = pd.DataFrame(columns=["mean_entries","sigy","sigx"])

        df_temp["mean_entries"] = means
        df_temp["sigy"] = [arr[0,1]] * len(means)
        df_temp["sigx"] = [arr[0,2]] * len(means)
    except:
        print("hdf name ", hdfname)
        print("arr sha ", arr.shape)
        print("df temp " , df_temp.shape)

    return df_temp


if __name__=="__main__":
    f=[]
    flag=True

    #run_names=["Dec11_2019","Dec19_2019","Nov20_2019","Nov21_2019","Nov22_2019","Nov23_2019","Nov25_2019","Nov27_2019","Nov29_2019"]

    #the following are only for validation
    run_names =["Dec2_2019", "Dec4_2019","Dec7_2019" ]

    hdflist=[]
    f=[]
    means=[]
    df_means = pd.DataFrame(columns=["mean_entries","sigy","sigx"])
    for run_name in run_names:
        p = Path.home()/"work/datasets/all_Runs/Sigx_Sigy/entries_romea"/run_name

        for hdf in p.rglob("*.h5"):
            df = pd.read_hdf(hdf, key="df")
            df = df[["entries","x_val","y_val"]]


            _,hdfname = os.path.split(str(hdf))

            df_means=get_means(df)
            f.append(df_means)


    #print("hdflist len ", len(hdflist))
    df = pd.concat(f,axis=0).reset_index()
    print("df shape outloop ", df.shape)

    #df.to_hdf("train_means_entries_all_sigx.h5",key="df")
    df.to_hdf("validation_means_entries_all_sigx.h5",key="df")



    #df = df.groupby(["x_val"],as_index=True, sort=False)[["entries", "y_val","x_val"]].apply(test).reset_index()
    #df = df.groupby(["sigx"],as_index=True, sort=False)[["mean_entries", "sigy","sigx"]].apply(plotter).reset_index()
    #df = df.groupby(["sigx"],as_index=True, sort=False)[["mean_entries", "sigy","sigx"]]
    print(df.columns)
    #print("df shape after group ", df.shape)



