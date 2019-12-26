import pandas as pd
import os, shutil,re
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import date


class ArrangeData:

    def __init__(self,imdir,Run_name): #imdir has to be in pathlib format

        self.imdir = imdir/Run_name
        self.Run_name = Run_name

    def iter_sigy(self, sigy_val):
        #sig_vals=[1.0,2.0,3.0,4.0]
        sig_vals = [sigy_val]
        for sig_val in sig_vals:
            for image in self.imdir.rglob("*.npy"):
                stem = image.stem
                sigy= image.stem.split('_')[2]
                temp_sigy_val = float(next(filter(str.isdigit, sigy)))
                if temp_sigy_val==sig_val:
                    sigy_val= temp_sigy_val
                    print(sigy_val)
                break


    def iter_sigx(self):
        sig_vals=[0.8, 1.0, 1.2, 1.4, 1.6]
        #sig_vals=[1.0,2.0,3.0,4.0]
        for sig_val in sig_vals:
            for image in self.imdir.rglob("*.npy"):
                stem = image.stem
                sigx= image.stem.split('_')[1]
                sigy= image.stem.split('_')[2]
                temp_sigx_val = float(next(filter(str.isdigit, sigx)))
                #temp_sigy_val = float(next(filter(str.isdigit, sigy)))
                if temp_sigx_val==sig_val:
                    sigx_val= temp_sigx_val
                    print(sigx_val)
                    self.iter_sigy(sigy_val)
                break


    def create_siglist(self):
        siglist =[]
        for i,image in enumerate(self.imdir.rglob("*.npy")):
            #print(image)
            stem = image.stem
            sigx= image.stem.split('_')[1]
            sigy= image.stem.split('_')[2]
            sig = sigx+"_"+sigy
            #print(sig)
            #print("in siglist")
            #if i==3:
            #    break
            #print("i ",i)
            if not sig in siglist:
                siglist.append(sig)
        print("siglist len ", len(siglist))
        return siglist




    def arrange_data(self): #this is the chief function

        siglist = self.create_siglist()
        sigset = set(siglist)

        #if len(sigset)<25:
        #    return 0

        sigitems=[]
        for i,sigdir in enumerate(sigset):
            #print(sigdir)
            tmplist = []
            for j,image in enumerate(self.imdir.rglob("*.npy")):

                stem = image.stem
                sigx= image.stem.split('_')[1]
                sigy= image.stem.split('_')[2]
                sig = sigx+"_"+sigy

                if sig==sigdir:
                     tmplist.append(str(image))
                     #print(len(tmplist))
                     #print(tmplist[0])
                     #break

            #print(tmplist)
            sigitems.append(tmplist)
            #break
            #if i==2:
            #    print(sigitems)
            #    break

        print(len(sigitems))

        df_dict = self.create_hdf(sigitems,sigset)

        return df_dict
        #return sigitems


    @staticmethod
    def ret_sigitems(sigitems): #fills missing values with None

        max_len = max(len(item) for item in sigitems)
        print(max_len)

        for item in sigitems:
            while len(item)<max_len:
                item.append(None)


        print("after ",len(sigitems))
        sigitems = np.array(sigitems)
        print(sigitems.shape)

        return sigitems


    def create_hdf(self,sigitems,sigset):

        sigitems=self.ret_sigitems(sigitems)
        df = pd.DataFrame(columns = sigset)

        for i,col in enumerate(df.columns):
            df[col]= sigitems[i,:]

        print("df shape before ", df.shape)
        df = df.dropna()
        print("df shape after dropna ", df.shape)

        df_train, df_test = train_test_split(df, test_size=0.1, train_size= 0.9)

        print("df shape after ", df.shape)
        print(df.head())
        print("train size ", df_train.shape)
        print("test size ", df_test.shape)


        p = Path.home()/"numpy_cnn"
        hdfdir =str(p)+"/"+ "hdfs"+"_"+str(date.today())
        if not os.path.exists(hdfdir):
            os.mkdir(hdfdir)


        hdfname = hdfdir+"/"+"train_"+self.Run_name+".h5"
        df_train.to_hdf(hdfname, key="df")
        print(hdfname+" is created")


        hdfname = hdfdir+"/"+"test_"+self.Run_name+".h5"
        df_test.to_hdf(hdfdir+"/"+"test_"+self.Run_name+".h5", key="df")
        print(hdfname+" is created")

        return  {"train":df_train, "test":df_test}



if __name__ == "__main__":

    imdir = Path.home()/"numpy_cnn/array_tar_new"
    Run_name="Nov22_2019"
    run_names = [x for x in os.listdir(str(imdir)) if re.search(r"\_",x) if x.split("_")[-1]=="2019"]
    #run_names.remove("Dec19_2019")

    f_train=[]
    f_test=[]
    for run_name in run_names:
        Arrange1 = ArrangeData(imdir, run_name)
        df_dict = Arrange1.arrange_data()

        if isinstance(df_dict, int):#make sure
            print("skipping ", run_name)

        elif isinstance(df_dict, dict):

            f_train.append(df_dict["train"])
            f_test.append(df_dict["test"])

        #f_train.append(df_dict["train"][0].reset_index(drop=True, inplace=True))
        #f_test.append(df_dict["test"][0].reset_index(drop=True, inplace=True))



    #df_train_all = pd.concat(f_train, axis=0)
    #try:
    df_train_all = pd.concat(f_train, axis=0,sort=False)
    df_train_all = df_train_all.dropna()
    #except:
    print("f_tran ", len(f_train))
    #    print("f_tran element shape", f_train[0].shape)
    #    print("ftrain ", ftrain[0])
    #    #print("df train all shape ", df_train_all.shape)

    df_test_all = pd.concat(f_test, axis=0,sort=False)
    df_test_all = df_test_all.dropna()
    print("df train all shape ", df_train_all.shape)
    #create_siglist()
    p = Path.home()/"numpy_cnn"
    hdfdir =str(p)+"/"+ "hdfs"+"_"+str(date.today())
    hdfname=hdfdir+"_all_train"+".h5"
    if not os.path.exists(hdfdir):
        os.mkdir(hdfdir)

    df_train_all.to_hdf(hdfname,key="df")
    print(hdfname+" is created")


    hdfname=hdfdir+"_all_test"+".h5"
    df_test_all.to_hdf(hdfname,key="df")
    print(hdfname+" is created")


