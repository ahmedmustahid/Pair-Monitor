import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import uproot
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import matplotlib.ticker as ticker
import os, shutil
import conf
import util
import numpy as np
#from util_plot import DfToNp, getXYZ
#from ROOT import TLorentzVector, TVector3
from util_plot import DfToNp, getXYZ
import conf
from util import getRunInfoList, makeDirectory
import glob, math
#from pathlib import Path

def set_axs(axs,title=None,xlabel=None,ylabel=None,textstr=None):
    #axs.set_title(title)

    axs.grid(True)
    axs.xaxis.set_minor_locator(AutoMinorLocator())
    axs.yaxis.set_minor_locator(AutoMinorLocator())
    axs.tick_params(which='both', width=1)
    axs.tick_params(which='major', length=7)
    axs.tick_params(which='minor', length=4)

    #textstr = 'entries: '+str(x.shape[0])
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.1)
    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)

    # place a text box in upper left in axes coords
    axs.text(0.75, 0.75, textstr, transform=axs.transAxes, fontsize="x-small", verticalalignment='top', bbox=props)
    #axs.text(0.65, 0.65, textstr, transform=axs.transAxes, fontsize=10, verticalalignment='top', bbox=props)


    axs.set_xlabel(xlabel, fontsize="x-large")
    axs.set_ylabel(ylabel, fontsize="x-large")

    #axs.set_xlim(left=30, right=None)

def iter_files_bunches(filenamelist,bunchnum):
    frames=[]
    for i,filename in enumerate(filenamelist):
        if i==bunchnum:
            break
        tree = uproot.open(filename)['evtdata_cont']
        df = tree.pandas.df(['BeamCal*'])
        #df = df.query('BeamCal_pdgcont==-11 and BeamCal_contz>0')
        df = df.query('BeamCal_contz>0')
        #df = df['BeamCal_contro']
        frames.append(df)
    df = pd.concat(frames)
    return df



def iter_files(filenamelist):
    frames=[]
    for i,filename in enumerate(filenamelist):
        if i==49:
            break
        #if i==11:
        #    print("opening ", filename)
        print i
        tree = uproot.open(filename)['evtdata_cont']
        #print "error inside ", filename," skipping"
        df = tree.pandas.df(['BeamCal*'])
        #df = df.query('BeamCal_pdgcont==-11 and BeamCal_contz>0')
        df = df.query('BeamCal_contz>0')
        #df = df['BeamCal_contro']
        frames.append(df)
    df = pd.concat(frames)
    return df


def root_to_hdf_different_bunches():
    runinfolist= getRunInfoList()
    bunchnums= np.linspace(1,50,10)

    #fig = plt.figure(figsize=(10,8))
    #colors = plt.rcParams["axes.prop_cycle"]()
    #ax = fig.add_subplot(111)

    #plotroots = []

    textstr =""
    hdflist = []

    for bunchnum in bunchnums:
        bunchnum = int(bunchnum)
        for j,runinfo in enumerate(runinfolist):
            dirname ="../"+ runinfo.getDirName_grid()+"/"+"root"+"/"+conf.Run_name
            #print(dirname)
            rootlist= [rootfile for rootfile in glob.glob(dirname+"/*.root")]
            filename=rootlist[0]
            #print rootlist[0]

            name,_=os.path.splitext(filename)
            name,_=os.path.splitext(name)
            _,name = os.path.split(name)

            name = name.split('_')[1]
            #print(name)

            hdfpath = "../"+ runinfo.getDirName_grid()+"/"+"hdf_files"
            makeDirectory(hdfpath)
            hdfname = hdfpath+"/"+runinfo.getDirName_grid()+"_"+str(bunchnum)+"_bunches_"+conf.Run_name+".h5"

            if not os.path.exists(hdfname):
                df = iter_files_bunches(rootlist, bunchnum)

                df.to_hdf(hdfname,key='df')

                print(hdfname+" created")
                hdflist.append(hdfname)
            else:
                print hdfname, " exists"
                hdflist.append(hdfname)

        print "hdf length ", len(hdflist)



def root_to_hdf_50_bunches():
    runinfolist= getRunInfoList()

    #fig = plt.figure(figsize=(10,8))
    #colors = plt.rcParams["axes.prop_cycle"]()
    #ax = fig.add_subplot(111)

    #plotroots = []

    textstr =""
    hdflist = []
    for j,runinfo in enumerate(runinfolist):
        dirname ="../"+ runinfo.getDirName_grid()+"/"+"root"+"/"+conf.Run_name
        #print(dirname)
        rootlist= [rootfile for rootfile in glob.glob(dirname+"/*.root")]
        #print(len(rootlist))
        filename=rootlist[0]
        print rootlist[0]

        name,_=os.path.splitext(filename)
        name,_=os.path.splitext(name)
        _,name = os.path.split(name)

        name = name.split('_')[1]
        #print(name)

        hdfpath = "../"+ runinfo.getDirName_grid()+"/"+"hdf_files"
        makeDirectory(hdfpath)
        hdfname = hdfpath+"/"+runinfo.getDirName_grid()+"_50_bunches_"+conf.Run_name+".h5"

        if not os.path.exists(hdfname):
            df = iter_files(rootlist)

            df.to_hdf(hdfname,key='df')

            print(hdfname+" created")
            hdflist.append(hdfname)
        else:
            print hdfname, " exists"
            hdflist.append(hdfname)

    print "hdf length ", len(hdflist)


class PhiPlotAllInOneWithArg:
#def phi_plot_all_in_one_with_arg(filenamelist, x0_y1, var_name, bunchnum=50,pdg="pos"):

    def __init__(self, filenamelist, x_or_y, var_name, bunchnum=50,pdg="pos"):

        self.filenamelist = filenamelist
        self.x_or_y = x_or_y
        self.var_name = var_name
        self.bunchnum = bunchnum
        self.pdg = pdg

    def plot(self):

        runinfolist= getRunInfoList()

        fig = plt.figure(figsize=(10,8))
        colors = plt.rcParams["axes.prop_cycle"]()
        ax = fig.add_subplot(111)

        plotroots = []

        textstr =""
        for j,hdfname in enumerate(self.filenamelist):

            name,_=os.path.splitext(hdfname)
            _,name = os.path.split(name)
            name = name.split('_')

            if self.x_or_y=="x":
                name = name[2] #name of non constant parameter
            elif self.x_or_y=="y":
                name = name[1]
            print(name)

            df = pd.read_hdf(hdfname,key="df")
            if self.pdg=="pos":
                df = df.query('BeamCal_pdgcont==-11')
            if not self.var_name=="energy":
                df = df['BeamCal_cont'+self.var_name]
                ar = DfToNp(df)
                textstr =textstr+"\n"+name+'entries:'+ str(ar.shape[0])

            if self.var_name=="phi":
                set_axs(ax,xlabel=r"$\phi$",ylabel='Number',textstr=textstr)

            elif self.var_name=="ro":
                set_axs(ax,xlabel=r'$\rho$',ylabel='Number',textstr=textstr)

            elif self.var_name=="energy":
                df = df['BeamCal_energycont']
                ar = DfToNp(df)
                textstr =textstr+"\n"+name+'entries:'+ str(ar.shape[0])
                ax.set_xlim((0,0.001))
                set_axs(ax,xlabel='energy',ylabel='Number',textstr=textstr)


            c = next(colors)["color"]
            ax.hist(ar,bins=100,histtype='step',label=name,density=False, color=c,log=False)
            #ax.hist(phi,bins=100,histtype='step',label=name,density=False, color=c,log=True)
            ax.legend(loc='upper right', ncol=2, framealpha=0.1)
            #ax.remove()


        newdir=conf.Run_name+"_plots"
        makeDirectory(newdir)
        bunchnum = self.bunchnum
        if var_name=="phi":
            if self.x_or_y=="x":
                if self.pdg=="pos":
                    plt.suptitle(r"$\phi$ Plots BeamCal_z>0 Positron "+str(bunchnum)+" bunches")
                    imname = newdir+"/"+ "phi_plots_"+str(bunchnum)+"bunch_log_Positron_allinone_const_x"
                else:
                    plt.suptitle(r"$\phi$ Plots BeamCal_z>0 "+str(bunchnum)+" bunches")
                    imname = newdir+"/"+ "phi_plots_"+str(bunchnum)+"bunch_log_allinone_const_x_nopdg"
            elif self.x_or_y=="y":
                if self.pdg=="pos":
                    plt.suptitle(r"$\phi$ Plots BeamCal_z>0 Positron "+str(bunchnum)+" bunches")
                    imname = newdir+"/"+ "phi_plots_"+str(bunchnum)+"bunch_log_Positron_allinone_const_y"
                else:
                    plt.suptitle(r"$\phi$ Plots BeamCal_z>0 "+str(bunchnum)+" bunches")
                    imname = newdir+"/"+ "phi_plots_"+str(bunchnum)+"bunch_log_allinone_const_y_nopdg"


        elif self.var_name=="ro":
            if self.x_or_y=="x":
                if self.pdg=="pos":
                    plt.suptitle(r"$\rho$ Plots BeamCal_z>0 Positron "+str(bunchnum)+" bunches")
                    imname = newdir+"/"+ "ro_plots_"+str(bunchnum)+"bunch_log_Positron_allinone_const_x"
                else:
                    plt.suptitle(r"$\rho$ Plots BeamCal_z>0 "+str(bunchnum)+" bunches")
                    imname = newdir+"/"+ "ro_plots_"+str(bunchnum)+"bunch_log_allinone_const_x_nopdg"

            elif self.x_or_y=="y":
                if self.pdg=="pos":
                    plt.suptitle(r"$\rho$ Plots BeamCal_z>0 Positron "+str(bunchnum)+" bunches")
                    imname = newdir+"/"+ "ro_plots_"+str(bunchnum)+"bunch_log_Positron_allinone_const_y"
                else:
                    plt.suptitle(r"$\rho$ Plots BeamCal_z>0 "+str(bunchnum)+" bunches")
                    imname = newdir+"/"+ "ro_plots_"+str(bunchnum)+"bunch_log_Positron_allinone_const_y_nppdg"



        elif self.var_name=="energy":
            if self.x_or_y=="x":
                if self.pdg=="pos":
                    plt.suptitle("Energy Plots BeamCal_z>0 Positron "+str(bunchnum)+" bunches")
                    imname = newdir+"/"+ "energy_plots_"+str(bunchnum)+"bunch_log_Positron_allinone_const_x"
                else:
                    plt.suptitle("Energy Plots BeamCal_z>0  "+str(bunchnum)+" bunches")
                    imname = newdir+"/"+ "energy_plots_"+str(bunchnum)+"bunch_log_Positron_allinone_const_x_nopdg"

            elif self.x_or_y=="y":
                if self.pdg=="pos":
                    plt.suptitle("Energy Plots BeamCal_z>0 Positron "+str(bunchnum)+" bunches")
                    imname = newdir+"/"+ "energy_plots_"+str(bunchnum)+"bunch_log_Positron_allinone_const_y"
                else:
                    plt.suptitle("Energy Plots BeamCal_z>0  "+str(bunchnum)+" bunches")
                    imname = newdir+"/"+ "energy_plots_"+str(bunchnum)+"bunch_log_Positron_allinone_const_y_nopdg"

        plt.savefig(imname+".png")
        plt.close()
        print imname+".png created"




def phi_plot_all_in_one(var_name, bunchnum=50):
    runinfolist= getRunInfoList()

    fig = plt.figure(figsize=(10,8))
    colors = plt.rcParams["axes.prop_cycle"]()
    ax = fig.add_subplot(111)

    plotroots = []

    textstr =""
    for j,runinfo in enumerate(runinfolist):
        dirname ="../"+ runinfo.getDirName_grid()+"/"+"root"+"/"+conf.Run_name

        hdfpath = "../"+ runinfo.getDirName_grid()+"/"+"hdf_files"
        makeDirectory(hdfpath)
        hdfname = hdfpath+"/"+runinfo.getDirName_grid()+"_"+str(bunchnum)+"_bunches_"+conf.Run_name+".h5"

        name,_=os.path.splitext(hdfname)
        _,name = os.path.split(name)


        name = name.split('_')

        name = name[1]+"_"+name[2]
        print(name)

        df = pd.read_hdf(hdfname,key="df")
        df = df.query('BeamCal_pdgcont==-11')

        if var_name=="phi":
            df = df['BeamCal_contphi']
            phi = DfToNp(df)
            textstr =textstr+"\n"+name+'entries:'+ str(phi.shape[0])
            set_axs(ax,xlabel=r'$\phi$',ylabel='Number',textstr=textstr)

        elif var_name=="ro":
            df = df['BeamCal_contro']
            phi = DfToNp(df)
            textstr =textstr+"\n"+name+'entries:'+ str(phi.shape[0])
            set_axs(ax,xlabel=r'$\rho$',ylabel='Number',textstr=textstr)

        elif var_name=="energy":
            df = df['BeamCal_energycont']
            phi = DfToNp(df)
            textstr =textstr+"\n"+name+'entries:'+ str(phi.shape[0])
            set_axs(ax,xlabel='energy',ylabel='Number',textstr=textstr)


        c = next(colors)["color"]

        #textstr =textstr+"\n"+name+'entries:'+ str(phi.shape[0])
        #set_axs(ax,xlabel=r'$\rho$',ylabel='Number',textstr=textstr)


        ax.hist(phi,bins=100,histtype='step',label=name,density=False, color=c,log=True)
        ax.legend(loc='upper right', ncol=3, framealpha=0.1)

    if var_name=="phi":
        plt.suptitle(r"$\phi$ Plots BeamCal_z>0 Positron")
        newdir=conf.Run_name+"_plots"
        makeDirectory(newdir)
        imname = newdir+"/"+ "phi_plots_"+str(bunchnum)+"bunch_log_Positron_allinone"


    elif var_name=="ro":
        plt.suptitle(r"$\rho$ Plots BeamCal_z>0 Positron")
        newdir=conf.Run_name+"_plots"
        makeDirectory(newdir)
        imname = newdir+"/"+ "ro_plots_"+str(bunchnum)+"bunch_log_Positron_allinone"

    elif var_name=="energy":
        plt.suptitle("Energy Plots BeamCal_z>0 Positron")
        newdir=conf.Run_name+"_plots"
        makeDirectory(newdir)
        imname = newdir+"/"+ "energy_plots_"+str(bunchnum)+"bunch_log_Positron_allinone"

    plt.tight_layout()

    plt.savefig(imname+".png")
    print imname+".png created"



def plot_all_one_param_constant(x_or_y, var_name, pdg="pos"):

    runinfolist= getRunInfoList()

    #fig = plt.figure(figsize=(10,8))
    #colors = plt.rcParams["axes.prop_cycle"]()
    #ax = fig.add_subplot(111)

    #plotroots = []

    textstr =""
    runlist = []
    sigxlist = []
    sigylist = []
    for j,runinfo in enumerate(runinfolist):

        runlist.append(runinfo.getDirName_grid())
        runname = runinfo.getDirName_grid()

        sigx = runname.split('_')[1]
        sigy = runname.split('_')[2]

        sigxlist.append(sigx)
        sigylist.append(sigy)


    sigxlist = set(sigxlist)
    sigylist = set(sigylist)

    hdfx_const = []
    hdfy_const = []


    for sigxval, sigyval in zip(sigxlist,sigylist):
        for j,runinfo in enumerate(runinfolist):


            dirname ="../"+ runinfo.getDirName_grid()+"/"+"root"+"/"+conf.Run_name
            #print(dirname)

            hdfpath = "../"+ runinfo.getDirName_grid()+"/"+"hdf_files"
            makeDirectory(hdfpath)
            hdfname = hdfpath+"/"+runinfo.getDirName_grid()+"_50_bunches_"+conf.Run_name+".h5"

            name,_=os.path.splitext(hdfname)
            #name,_=os.path.splitext(name)
            _,name = os.path.split(name)


            name = name.split('_')

            #every 4 elements of the list constant parameter
            if name[2]==sigyval:
                hdfy_const.append(hdfname)
                #print "y const hdf ",hdfname
            if name[1]==sigxval:
                hdfx_const.append(hdfname)


    if x_or_y=="y":
        for hdfy in hdfy_const:
            print("hdfyconst ")
            print(hdfy)
        phiplot = PhiPlotAllInOneWithArg(hdfy_const[5:10], x_or_y, var_name, bunchnum=50)
        phiplot.plot()

    elif x_or_y=="x":
        for hdfx in hdfx_const:
            print("hdfxconst ")
            print(hdfx)
        phiplot = PhiPlotAllInOneWithArg(hdfx_const[:5], x_or_y, var_name, bunchnum=50)
        phiplot.plot()




def plot_all_one_param_constant_bunch(x_or_y, var_name, bunchnum):

    runinfolist= getRunInfoList()

    #fig = plt.figure(figsize=(10,8))
    #colors = plt.rcParams["axes.prop_cycle"]()
    #ax = fig.add_subplot(111)

    #plotroots = []

    textstr =""
    runlist = []
    sigxlist = []
    sigylist = []
    for j,runinfo in enumerate(runinfolist):

        runlist.append(runinfo.getDirName_grid())
        runname = runinfo.getDirName_grid()

        sigx = runname.split('_')[1]
        sigy = runname.split('_')[2]

        sigxlist.append(sigx)
        sigylist.append(sigy)


    sigxlist = set(sigxlist)
    sigylist = set(sigylist)

    hdfx_const = []
    hdfy_const = []

    for sigxval, sigyval in zip(sigxlist,sigylist):
        for j,runinfo in enumerate(runinfolist):


            dirname ="../"+ runinfo.getDirName_grid()+"/"+"root"+"/"+conf.Run_name
            #print(dirname)

            hdfpath = "../"+ runinfo.getDirName_grid()+"/"+"hdf_files"
            makeDirectory(hdfpath)
            hdfname = hdfpath+"/"+runinfo.getDirName_grid()+"_"+str(bunchnum)+"_bunches_"+conf.Run_name+".h5"

            name,_=os.path.splitext(hdfname)
            #name,_=os.path.splitext(name)
            _,name = os.path.split(name)


            name = name.split('_')

            #every 4 elements of the list constant parameter
            if name[2]==sigyval:
                hdfy_const.append(hdfname)
                #print "y const hdf ",hdfname
            if name[1]==sigxval:
                hdfx_const.append(hdfname)

    if x_or_y=="y":
        for hdfy in hdfy_const:
            print hdfy
        phiplot = PhiPlotAllInOneWithArg(hdfy_const[4:8], y1, var_name, bunchnum)
        phiplot.plot()

    elif x_or_y=="x":
        for hdfx in hdfx_const:
            print hdfx
        phiplot = PhiPlotAllInOneWithArg(hdfx_const[:4], x0, var_name, bunchnum)
        phiplot.plot()





if __name__ == "__main__":

    var_name="ro" #"energy", "ro", "phi"
    plot_all_one_param_constant(x_or_y="x",var_name=var_name, pdg="")
    plot_all_one_param_constant(x_or_y="y",var_name=var_name, pdg="")

    var_name="phi" #"energy", "ro", "phi"
    plot_all_one_param_constant(x_or_y="x",var_name=var_name, pdg="")
    plot_all_one_param_constant(x_or_y="y",var_name=var_name, pdg="")

    var_name="energy" #"energy", "ro", "phi"
    plot_all_one_param_constant(x_or_y="x",var_name=var_name, pdg="")
    plot_all_one_param_constant(x_or_y="y",var_name=var_name, pdg="")




