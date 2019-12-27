import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd

from pathlib import Path

import seaborn as sns
import sys
from sklearn.svm import SVC
from sklearn.metrics import classification_report, plot_roc_curve,  plot_confusion_matrix
from sklearn.utils import shuffle


def create_features_target(df):

    df = df.copy()
    print(df.columns)

    le = preprocessing.LabelEncoder()
    df['label'] = le.fit_transform(df['sigy'])

    print("df label ", df["label"].shape)
    print("df mean ", df["mean_entries"].shape)

    #clf = SVC()
    X = df["mean_entries"].to_numpy()
    y = df["label"].to_numpy()

    class_names = le.inverse_transform(y)

    print("y shape ", y.shape)
    print("x shape ", X.shape)
    print("x reshape ", X.reshape(-1,1).shape)

    return X.reshape(-1,1),y,class_names



if __name__=="__main__":

    ptrain=Path.cwd()/"train_means_entries_all_sigx.h5"
    pvalidation=Path.cwd()/"validation_means_entries_all_sigx.h5"

    train_df = pd.read_hdf(ptrain, key="df")
    validation_df = pd.read_hdf(pvalidation, key="df")

    #GET CONSTANT SIGX MEAN GROUP
    train_dfs_wth_sigx = [(sigx,df_sigy) for sigx,df_sigy in train_df.groupby("sigx") ] #GROUPBY OUTPUTS GROUP VALUE and DATAFRAME
    val_dfs_wth_sigx = [(sigx,df_sigy) for sigx,df_sigy in validation_df.groupby("sigx") ]


    for (sigxt,dft),(sigxv,dfv) in zip(train_dfs_wth_sigx, val_dfs_wth_sigx):
        print(sigxt)

        X_train, y_train,_ = create_features_target(dft)

        dfv = shuffle(dfv)
        X_val, y_val,class_names = create_features_target(dfv)
        clf = SVC()
        clf.fit(X_train, y_train)


        y_pred = clf.predict(X_val.reshape(-1,1))
        print("svm score ", clf.score(X_val, y_val))

        if isinstance(class_names,list):
            class_names=np.array(class_names)
        class_names = np.unique(class_names)
        plot_confusion_matrix(clf, X_val, y_val,display_labels=class_names,cmap=plt.cm.Blues)
        plt.savefig("confusion_sigy_class.png")

        break











