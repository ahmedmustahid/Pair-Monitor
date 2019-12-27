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


class CreateEMD:

    def __init__(self,run_names,sigx_vals,sections=5):

        self.run_names=run_names
        self.sections = sections
        self.sigx_vals = sigx_vals


    @staticmethod
    def plotter(df):
        sns.violinplot(x="sigy",y="mean_entries",data=df)
        plt.xlabel(r"$\sigma_y$",fontsize="x-large")
        plt.ylabel(r"$Mean_{entries}$",fontsize="x-large")
        plt.suptitle(r"$\sigma_x$ constant: "+str(df.iloc[0,2]))

        plotdir = "plots"+"_"+str(date.today())
        if not os.path.exists(plotdir):
            os.mkdir(plotdir)
        plotname=plotdir+ "/all_entries_sigx_"+str(df.iloc[0,2])+"_"+str(date.today())+".png"
        plt.savefig(plotname)
        print(plotname+" is created")
        plt.close()

    @staticmethod
    def get_minshape(path):#path will be in pathlib format
        if not isinstance(path,pathlib.PosixPath): #"Only PosixPath arguument"
           raise TypeError("only pathlib.PosixPath as argument.")
        templen =[]
        for i,hdf in enumerate(path.rglob("*.h5")):
            df = pd.read_hdf(hdf,key="df")
            arr = df[["entries","y_val","x_val"]].to_numpy()
            entries = arr[:,0]
            #print("entries shape ", arr[:,0].shape[0])
            templen.append(arr[:,0].shape[0])
        min_shape = min(templen)

    @staticmethod
    def get_all_entries(path):
        if not isinstance(path,pathlib.PosixPath): #"Only PosixPath arguument"
            raise TypeError("only pathlib.PosixPath as argument.")

            temp_entries=[]
            hdfnames=[]
            sigys=[]
            sigxs=[]
            for i,hdf in enumerate(p.rglob("*.h5")):
                df = pd.read_hdf(hdf,key="df")
                arr = df[["entries","y_val","x_val"]].to_numpy()
                entries = arr[:min_shape,0]
                sigy = arr[0,1]
                sigx = arr[0,2]

                head, tail = os.path.split(str(hdf))
                root, ext = os.path.splitext(tail)
                hdfnames.append(root)
                temp_entries.append(entries)
                sigys.append(sigy)
                sigxs.append(sigx)


    #def get_source(arr):
    #

    #    for i in arr.shape[0]:








    def get_means(self,df):

        arr = df.to_numpy()
        #print("arr shape ", arr.shape)
        sections=self.sections
        arrlist = np.array_split(arr, indices_or_sections=sections, axis=0)

        xs = self.get_source(arrlist[0])


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
            #print("hdf name ", hdfname)
            print("arr sha ", arr.shape)
            print("df temp " , df_temp.shape)

        return df_temp






    def iter_runs(self):

        hdflist=[]
        f=[]
        means=[]
        df_means = pd.DataFrame(columns=["mean_entries","sigy","sigx"])

        for run_name in self.run_names:
            p = Path.home()/"work/datasets/all_Runs/Sigx_Sigy/entries_romea"/run_name

            for hdf in p.rglob("*.h5"):
                df = pd.read_hdf(hdf, key="df")
                df = df[["entries","x_val","y_val"]]


                _,hdfname = os.path.split(str(hdf))

                df_means=self.get_means(df)
                f.append(df_means)

        df = pd.concat(f,axis=0).reset_index()
        print("df shape outloop ", df.shape)

        #self.create_hdf(df,self.str_type)
        hdfdir = "EMD_hdfs"+"_"+str(date.today())
        if not os.path.exists(hdfdir):
            os.mkdir(hdfdir)

        hdfname=hdfdir+"/"+"emds_all_sigx"+str(date.today())+".h5"
        df.to_hdf(hdfname,key="df")
        print(hdfname+" created")

        #PLOT FOR EACH GROUP of SIGX
        print(df.columns)
        df.groupby(["sigx"],axis=0, as_index=True,sort=False)["mean_entries","sigy","sigx"].apply(self.plotter)



def find_sd(df):

    emd = df.to_numpy()
    sd = np.std(emd)
    mean = np.mean(emd)
    return pd.Series({"sd":sd, "mean":mean})


def cal_midedges(x):

    if isinstance(x, list):
        x = np.array(x)

    edges_mid = np.zeros(x.shape[0]-1)
    for i in range(x.shape[0]-1):
        edges_mid[i] = (x[i]+x[i+1])/2
    return edges_mid


def cal_kantoro(xs,xt,bins=10):

    H, xs_edges, xt_edges = np.histogram2d(xs,xt,bins=bins)

    H/=H.sum()
    xs_wt = np.sum(H,axis=0)
    xt_wt = np.sum(H,axis=1)

    xs_pos = xs_edges[1:]
    xt_pos = xt_edges[1:]

    xs_pos = cal_midedges(xs_edges)
    xt_pos = cal_midedges(xt_edges)

    #cost matrix
    n=xs_pos.shape[0]
    M = ot.dist(xs_pos.reshape(n,1), xt_pos.reshape(n,1), metric='cityblock')

    G = ot.emd(xs_wt, xt_wt, M)

    emd = np.sum(np.multiply(G,M))
    return emd


if __name__=="__main__":

    df = pd.DataFrame(columns=["emd","sigx","sigy"])
    run_names=["Dec11_2019","Dec19_2019","Nov20_2019","Nov21_2019","Nov22_2019","Nov23_2019","Nov25_2019","Nov27_2019","Nov29_2019"]
    #run_names1 =["Dec2_2019", "Dec4_2019","Dec7_2019" ]
    #run_names += run_names1


    #for sigx_val in sigx_vals:
    sigx_vals = [0.8,1.0,1.2,1.4,1.6]

    for sigx_val in sigx_vals:
        f = []
        for count,run_name in enumerate(run_names):
            print("run ", run_name)
            p = Path.cwd()/".."/run_name
            temp_entries=[]
            hdfnames=[]
            sigys=[]
            sigxs=[]

            temp =[]
            for i,hdf in enumerate(p.rglob("*.h5")):
                df = pd.read_hdf(hdf,key="df")
                arr = df[["entries","y_val","x_val"]].to_numpy()
                entries = arr[:,0]
                #print("entries shape ", arr[:,0].shape[0])
                temp.append(arr[:,0].shape[0])
            min_shape = min(temp)

            for i,hdf in enumerate(p.rglob("*.h5")):
                df = pd.read_hdf(hdf,key="df")
                arr = df[["entries","y_val","x_val"]].to_numpy()
                entries = arr[:min_shape,0]
                sigy = arr[0,1]
                sigx = arr[0,2]

                head, tail = os.path.split(str(hdf))
                root, ext = os.path.splitext(tail)
                hdfnames.append(root)
                temp_entries.append(entries)
                sigys.append(sigy)
                sigxs.append(sigx)
            #    if i==1:
            #        break

            #print(len(temp_entries))
            #print(len(sigys))
            #print(sigys)

            for sigx,sigy,entries in zip(sigxs,sigys,temp_entries):
                if str(sigy)=="0.8" and str(sigx)=="0.8":
                #if str(sigy)=="1.0" and str(sigx)=="1.0":
                    #print("inside 1.0")
                    std_dist=entries
                    break



            sigy_dict =[]
            sigx_dict =[]
            emd_dict=[]
            for sigx,sigy,entries in zip(sigxs,sigys,temp_entries):

                #print("emd betn sigy1.0 and sigx{} sigy{} :".format(sigx,sigy))
                #print(hdf)
                xs = std_dist
                xt = entries
                emd = cal_kantoro(xs,xt,bins=100)
                sigy_dict.append(sigy)
                sigx_dict.append(sigx)
                emd_dict.append(emd)

            d = {"emd":emd_dict, "sigy":sigy_dict, "sigx":sigx_dict}
            df = pd.DataFrame.from_dict(d)


            #sigx_val=0.8
            df = df.loc[df["sigx"]==float(sigx_val)]
            #df = df.query("sigx==@sigx_val")

            f.append(df)


        df = pd.concat(f,axis=0).reset_index()


        sns.violinplot(x="sigy",y="emd",data=df)
        plt.suptitle(r"$\sigma_x$ constant: "+str(sigx_val))
        plt.xlabel(r"$\sigma_y$", fontsize="x-large")
        #plt.show()
        plt.savefig("violin_sigx_"+str(sigx_val)+".png")
        plt.close()
        df.to_hdf("emd_sigx"+str(sigx_val)+".h5",key="df")

        series = df.groupby(["sigy"],as_index=True, sort=False)["emd"].apply(find_sd).reset_index()
        #series = df_scaled.groupby(["sigy"],as_index=True, sort=False)["emd"].apply(find_sd).reset_index()
        print(series)

        #a = df.to_numpy()
        #print(a)



