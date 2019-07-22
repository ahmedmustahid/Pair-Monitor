import numpy as np
import sys,os,optparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import uproot
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import matplotlib.ticker as ticker
from util_plot import getFromDf, set_axs, inFile
import os



def plot_args(df,args,fig,axs,fname):
    try:
           axs_iter = iter(axs)
    except TypeError:
           print(axs,'is not iterable')
    axs = axs.flatten()
    fig.suptitle(fname,fontsize = 25)
    for ax, arg in zip(axs, args):
        print('arg in plot_args',arg)
        x = getFromDf(df,arg)
        textstr='entries:'+str(x.shape[0])
        set_axs(ax,xlabel=arg,textstr=textstr)
        ax.hist(x,bins=50,log=True,histtype='step')
    plt.savefig(directory+'/'+'auto_plots_'+str(counter+1)+'.png')
    print(directory+'/'+'auto_plots_'+str(counter+1)+'.png files created')
    args.clear()

inFile, fname = inFile()

#inFile='df_wth_r.csv'
df = pd.read_csv(inFile, sep='|')

#directory = inFile.split('.')[0]

directory = fname
print(directory)
print(type(directory))

directory = directory+'_plots'
if not os.path.exists(directory):
    os.makedirs(directory)


df2 = df.filter(regex='BeamCal_*')
df3 = df.filter(regex='mcp_end*')
df4 = df.filter(regex='mcp_start*')
df5 = df[['mcp_e','mcp_px','mcp_py','mcp_pz']]

df = [df2, df3, df4, df5]
df = pd.concat(df,axis=1)

args=[]
print(df.columns)

n = 0
m = 0
for counter,col in enumerate(df.columns):
    args.append(col)
    print(counter)
    if (counter+1)%4==0 :
        fig, axs= plt.subplots(nrows=2,ncols=2,sharex=False, sharey=True,figsize=(15,18))
        plot_args(df,args,fig,axs,fname)
        n = n+1
        m =  df.columns.shape[0] - 4 * n
    elif(m ==2 or m==1 or m==3):
        print(col)
        fig, axs= plt.subplots(nrows=1,ncols= 1 ,sharex=False, sharey=True,figsize=(8,10))
        fig.suptitle(fname,fontsize = 25)
        x = getFromDf(df,str(col))
        textstr='entries:'+str(x.shape[0])
        set_axs(axs,xlabel=str(col),textstr=textstr)
        axs.hist(x,bins=50,log=True,histtype='step')
        plt.savefig(directory+'/'+'auto_'+col+'_'+str(counter+1)+'.png')
        print(directory+'/'+'auto_'+col+'_'+str(counter+1)+'.png is created')
