
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV

from pathlib import Path
import pandas as pd
import seaborn as sns
import sys
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, max_error, explained_variance_score, mean_absolute_error

#from FastBDT import Classifier
from PyFastBDT import FastBDT
import os

from datetime import date
import time
import sys
import conf


class PlotAndEval:

    def __init__(self, model, X_train, X_test, y_train, y_test,model_name="model"):

        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name= model_name



    @staticmethod
    def create_hist(y_test, y_pred,model_name):

        xy_colnums = [0, 1]
        index = np.arange(y_test.shape[0])
        y_test = y_test.set_index(index)
        siglist = [0.8,1.0, 1.2 , 1.4, 1.6]
        fig,ax = plt.subplots()

        #for colnum in xy_colnums:
        print("y_pred all shape ", y_pred.shape)
        colnum = 0
        #for colnum in xy_colnums:
        n = len(siglist)
        color=iter(cm.rainbow(np.linspace(0,1,n)))
        for sig,c in zip(siglist,color):

            y_pred = y_pred.reshape(-1,1)
            colnum=0

            #idx is the index of true values
            idx = y_test.index[y_test.iloc[:,colnum]==sig]
            #pred values corresponding to true ones are put into b
            b = y_pred[idx,colnum]

            ax.hist(b.reshape(-1,1), bins=100, log=True, alpha=0.5, label=str(sig), range=(0.2,2))
            #ax.hist(b.reshape(-1,1), bins=100, log=True, alpha=0.5, label=str(sig), range=(0.2,2),color=c)
            ticks = np.arange(0.8,1.8,step=0.2)
            ax.set_xticks(ticks)


            ax.set_xlabel(r"$\sigma_x$",fontsize="x-large")
            ax.set_title("Model: "+model_name+" Regression")

            #make sure axes dont repeat
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), framealpha=0.3)

        plotdir=Path.cwd()/"reg_plotdir"/conf.Run_name
        if not os.path.exists(str(plotdir)):
            os.mkdir(str(plotdir))
        plotname= str(plotdir)+"/"+model_name+"_"+str(colnum)+"_"+str(date.today())+"_"+conf.Run_name+".png"

        plt.savefig(plotname)
        print(plotname+" is created")

        plt.close("all")
        ax.remove()



    @staticmethod
    def get_errors(y_true, y_pred):

        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        expl_var = explained_variance_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        maxer = max_error(y_true, y_pred)

        return mse, r2, expl_var, mae, maxer




    def calc_times(self):

        start = time.time()
        model = self.model


        model.fit(self.X_train, self.y_train)

        if self.model_name=="Ridge":
            alpha= model.alpha_
            print("alpha: "+repr(alpha)+" "+self.model_name)


        y_pred = model.predict(self.X_test)

        #print("y_true")
        #print(self.y_test[:15])

        #print("pred")
        #print(y_pred[:15])

        #print("score=%.2f" % model.score(X_test, y_test))
        self.create_hist(self.y_test, y_pred,self.model_name )

        elapsed_time = time.time() - start
        mse,r2 , expl_var, mae, maxer= self.get_errors(self.y_test, y_pred)

        print("time required ", elapsed_time)

        return mse, r2, expl_var, mae, maxer




# Create a random dataset
rng = np.random.RandomState(1)
def create_data_label(data_path):

    data = pd.read_hdf(str(data_path), key="df")
    data = shuffle(data)
    #targets = data[["x_val", "y_val"]]
    targets = data[["x_val"]]
    features = data.drop(["x_val","y_val"], axis=1)
    return (features, targets)


if __name__=="__main__":
    p = Path.home()
    fname_train = conf.Run_name+"/combined_train_2019-12-19_"+conf.Run_name+".h5"
    train_path =  p/"numpy_cnn/regression/hdf_files/train"/fname_train

    fname_test =conf.Run_name+"/combined_test_2019-12-19_"+conf.Run_name+".h5"
    test_path = p/"numpy_cnn/regression/hdf_files/test"/fname_test
    X_test, y_test = create_data_label(test_path)
    X_train, y_train = create_data_label(train_path)



    estimators = {
        "K-nn": KNeighborsRegressor(n_neighbors=1, algorithm="kd_tree",leaf_size=30),
        "RandomForest": RandomForestRegressor(n_estimators=1000,max_depth=1000),
        "Linear": LinearRegression(),
        "Ridge": RidgeCV(alphas=[1,10,100]),
        "Lasso": LassoCV()
    }

    mses =[]
    r2s =[]
    expl_vars=[]
    maes=[]
    maxers=[]
    #modelnames=[]
    for est in estimators.keys():
        print("keys ", est)
        #print("values ", estimators[est])
        #CalcTimes(estimators[est], X_train, X_test, y_train, y_test,model_name=est)

        PlotterEval=PlotAndEval(estimators[est], X_train, X_test, y_train, y_test,model_name=est)
        mse, r2, expl_var, mae, maxer = PlotterEval.calc_times()

        mses.append(mse)
        r2s.append(r2)
        expl_vars.append(expl_var)
        maes.append(mae)
        maxers.append(maxer)



        #CalcTimes(model, X_train, X_test, y_train, y_test)

    df = pd.DataFrame(columns=["mse","r2","mae","expl_var","maxer","model_name"])
    df["mse"]=mses
    df["r2"]=r2s
    df["mae"] = maes
    df["maxer"] = maxers
    df["expl_var"]=expl_vars

    df["model_names"]=list(estimators)

    hdfdir = "errors_reg"
    if not os.path.exists(hdfdir):
        os.mkdir(hdfdir)
    df.to_hdf(hdfdir+"model_error.h5",key="df")




