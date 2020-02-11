import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import matplotlib.ticker as ticker
from datetime import date

import os,sys

class Plotter:

    def __init__(self,df,divnum,philims):

        self.df = df
        self.divnum=divnum
        self.philims=philims

    @staticmethod
    def set_axs(axs,title=None,xlabel=None,ylabel=None,textstr=None):
        #axs.set_title(title,fontsize="x-large")

        axs.grid(True)
        axs.xaxis.set_minor_locator(AutoMinorLocator())
        axs.yaxis.set_minor_locator(AutoMinorLocator())
        axs.tick_params(which='both', width=1)
        axs.tick_params(which='major', length=7)
        axs.tick_params(which='minor', length=4)

        #textstr = 'entries: '+str(x.shape[0])
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        axs.text(0.05, 0.95, textstr, transform=axs.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)


        axs.set_xlabel(xlabel,fontsize="x-large")
        #axs.set_ylabel(ylabel)


    def create_div(self):
        llim=self.philims[0]
        ulim=self.philims[1]
        #df1 = self.df.query("@llim<=phi<=@ulim")
        fig,ax = plt.subplots(figsize=(10,8))
        sigys=[0.8,1.0,1.2,1.4,1.6]

        for sigy in sigys:

            df2= df1.query("y_val==@sigy")
            phi = df2["phi"].to_numpy()
            np.random.shuffle(phi)
            arrs = np.array_split(phi,self.divnum)

            print("sigy ",sigy)
            for ar in arrs:
                ax.hist(ar, bins=100, label=r"$\sigma_y$:"+str(sigy),histtype="step")

            xlabel=r"$\phi$"
            title=str(llim)+r"$\leq \phi \leq$"+str(ulim)+" after "+str(self.divnum)+" division"+" "+r"$\sigma_y$:"+str(sigy)

            self.set_axs(ax,xlabel=xlabel,title=title)
            break

        imdir="many_divs"+str(date.today())
        if not os.path.exists(imdir):
            os.mkdir(imdir)

        plt.savefig(imdir+"/"+"test_"+str(self.divnum)+"_llim_"+str(llim)+"_ulim_"+str(ulim)+"_div_"+str(date.today())+".png")
        ax.remove()
        plt.close()

    def plot_allsigs(self):

        llim=self.philims[0]
        ulim=self.philims[1]
        #df1 = self.df.query("@llim<=phi<=@ulim")
        df1=self.df
        fig,ax = plt.subplots(figsize=(10,8))
        sigys=[0.8,1.0,1.2,1.4,1.6]

        for sigy in sigys:

            df2= df1.query("y_val==@sigy")
            phi = df2["phi"].to_numpy()

            xlabel=r"$\phi$"
            title=str(llim)+r"$\leq \phi \leq$"+str(ulim)
            self.set_axs(ax,xlabel=xlabel,title=title)

            ax.hist(phi, bins=100, label=r"$\sigma_y$:"+str(sigy), histtype="step")
            ax.legend()


        imdir="many_divs"+str(date.today())
        if not os.path.exists(imdir):
            os.mkdir(imdir)

        #plt.savefig(imdir+"/"+"test"+"_allsigs_"+"div_"+str(self.divnum)+"_llim_"+str(llim)+"_ulim_"+str(ulim)+"_"+str(date.today())+".png")
        plt.savefig(imdir+"/"+"test"+"_allsigs_"+"div_"+str(self.divnum)+"_llim_"+str(llim)+"_ulim_"+str(ulim)+"_"+str(date.today())+"_nocut.png")
        ax.remove()
        plt.close()


if __name__=="__main__":

    p= Path.cwd()/"combined.h5"
    df = pd.read_hdf(p, key="df")
    print(df.columns)

    divnum=10
    philims=[0,1]
    #philims=[0.625,0.75]

    P1 = Plotter(df,divnum,philims)
    #P1.create_div()
    P1.plot_allsigs()

    #divnums=np.arange(10,60,step=10)
    #for divnum in divnums:
    #    P1 = Plotter(df,divnum,philims)
    #    P1.create_div()
    #    P1.plot_allsigs()




