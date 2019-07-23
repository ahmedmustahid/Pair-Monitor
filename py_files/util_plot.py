import numpy as np
import sys,os,optparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import uproot
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import matplotlib.ticker as ticker
from ROOT import TLorentzVector, TVector3
import math
import shutil
def DfToNp(df):
    df = df.to_numpy()
    df = df.ravel()
    return df

def getXYZ(df,args):
    print(args[0]+' corresponds to x values')
    #print(df.columns)
    dfx= df[[args[0]]]
    dfx=dfx.dropna(axis=0)
    #print(dfx)
    print(dfx.shape)
    dfy= df[[args[1]]]
    dfy=dfy.dropna(axis=0)
    dfz= df[[args[2]]]
    dfz=dfz.dropna(axis=0)

    x=DfToNp(dfx)
    y=DfToNp(dfy)
    z=DfToNp(dfz)

    return (x,y,z)

def getXYZ1(df,args):
    print(args[0]+' corresponds to x values')
    #print(df.columns)
    dfx= df[[args[0]]]
    #dfx=dfx.dropna(axis=0)
    #print(dfx)
    print(dfx.shape)
    dfy= df[[args[1]]]
    #dfy=dfy.dropna(axis=0)
    dfz= df[[args[2]]]
    #dfz=dfz.dropna(axis=0)

    x=DfToNp(dfx)
    y=DfToNp(dfy)
    z=DfToNp(dfz)

    return (x,y,z)


def getFromDf(df,arg):
    df=df[[arg]]
    df = df.dropna(axis=0)
    x = DfToNp(df)
    return x

def set_axs(axs,title=None,xlabel=None,ylabel=None,textstr=None):
    axs.set_title(title)

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


    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)


def cal_r(x,y):
    r = np.square(x)+np.square(y)
    r = np.sqrt(r)
    return r

def cal_phi(x,y):
    phi = np.arctan2(y,x)
    return phi

def spherical2(x,y,z):
    ro = np.hypot(x,y)
    r = np.hypot(ro,z)
    theta = np.arctan2(ro,z)
    phi = np.arctan2(y,x)
    return (r,ro, theta, phi)

#def spherical(x,y,z):
#    ro = np.hypot(x,y)
#    r = np.hypot(ro,z)
#    theta = np.arctan2(ro,z)
#    phi = np.arctan2(y,x)
#    return (r, ro, theta, phi)
def spherical(x,y,z):
    with np.nditer([x,y,z,None,None,None,None]) as it:
        for i,j,k,r,ro,theta,phi in it:
            v = TVector3()
            v.SetXYZ(i,j,k)
            r[...] = v.Mag()
            ro[...] = math.hypot(i,j)
            theta[...] = v.Theta()
            phi[...] = v.Phi()
        return (it.operands[3],it.operands[4],it.operands[5],it.operands[6])

def plt_hist2d(x,y,bins,rang=None, title = None, xlabel=None, ylabel=None,fname = 'test'):
    plt.jet()
    fig = plt.figure()
    axs = fig.add_subplot(111)
    _,_,_,pcm=axs.hist2d(x=x,y=y,range=rang, bins =bins,norm=colors.LogNorm())
    fig.colorbar(pcm,ax=axs)
    textstr = 'entries: '+str(x.shape[0])
    set_axs(axs,title,xlabel,ylabel,textstr)
    plt.savefig(fname+'.png')

def getPltFile(df,args,fname,title='mcp_Geant'):
    x,y,z = getXYZ(df,args)

    r = cal_r(x,y)

    plt_hist2d(z,r,bins=50,title=title,xlabel='z(mm)',ylabel='r(mm)')

    plt.savefig(fname+'.png')

def getPltFile2(df,args,fname,rang,title='BCal_geant'):
    x,y,z = getXYZ(df,args)

    r = cal_r(x,y)

    plt_hist2d(z,r,bins=50,rang=rang,title=title,xlabel='z(mm)',ylabel='r(mm)')

    plt.savefig(fname+'.png')

def plt_hist(ax,x,bins=None, range=None, histtype='bar',log='True' ):
    ax.hist(x,bins=None, range=None, histtype='bar',log='True')

def inFile():
    usage = 'usage: %prog input.csv'
    parser = optparse.OptionParser(usage=usage)
    (options, args) = parser.parse_args()
    if(len(args)) != 1:
        parser.error('Please enter csv file name')

    inFile=args[0]
    fname = inFile.split('.')
    if ('/' in fname):
        fname = fname.split('/')[1]

    if(len(fname)==3):
        fname = fname[0]+'.'+fname[1]

    return (inFile,fname)

def boost(df,args): #args must have 4 elements xyzt or pxpypzE
    px,py,pz = getXYZ(df,args)
    E = DfToNp(df[[args[3]]].dropna(axis=0))
    l = TLorentzVector()
    with np.nditer([px,py,pz,E,None,None,None,None]) as it:
        for pi,pj,pk,el,pm,en,phi,cost in it:
            l.SetPxPyPzE(pi,pj,pk,el)
            bi = pi/el
            bj = pj/el
            bk = pk/el
            l.Boost(-bi,-bj,-bk)
            pm[...]= l.Px()
            en[...] = l.E()
            phi[...] = l.Phi()
            cost[...] = l.CosTheta()
        return (it.operands[4],it.operands[5],it.operands[6],it.operands[7])

def momentum_aftr_boost(df,args): #args must have 4 elements xyzt or pxpypzE
    px,py,pz = getXYZ(df,args)
    E = DfToNp(df[[args[3]]].dropna(axis=0))
    l = TLorentzVector()
    with np.nditer([px,py,pz,E,None,None,None,None]) as it:
        for pi,pj,pk,el,pm,pn,po,eq in it:
            l.SetPxPyPzE(pi,pj,pk,el)
            bi = pi/el
            l.Boost(-bi,0,0)
            pm[...]= l.Px()
            pn[...] = l.Py()
            po[...] = l.Pz()
            eq[...] = l.E()
        return (it.operands[4],it.operands[5],it.operands[6],it.operands[7])

def position_aftr_boost(df,args): #args must have 4 elements xyzt or pxpypzE
    px,py,pz = getXYZ(df,args)
    E = DfToNp(df[[args[3]]].dropna(axis=0))
    l = TLorentzVector()
    with np.nditer([px,py,pz,E,None,None,None]) as it:
        for pi,pj,pk,el,x,y,z in it:
            l.SetPxPyPzE(pi,pj,pk,el)
            #p = math.hypot(pi,pj)
            #p = math.hypot(pk,p)
            bi = pi/el
            bj = pj/el
            bk = pk/el
            #print(b)
            l.Boost(-bi,-bj,-bk)
            z[...]= l.X()
            y[...] = l.Y()
            z[...] = l.Z()
        return (it.operands[4],it.operands[5],it.operands[6])
"""
def add_sphere_param():
    inFile, fname = inFile()
    df = pd.read_csv(inFile,sep='|')
    df = df.dropna()
    args = ['mcp_endx', 'mcp_endy','mcp_endy']
    x,y,z = getXYZ(df,args)
    r, ro, theta, phi = spherical(x,y,z)
    df['mcp_endr'] = r
    df['mcp_endro'] = ro
    df['mcp_endtheta'] = theta
    df['mcp_endphi'] = phi


    args = ['BeamCal_x', 'BeamCal_y','BeamCal_y']
    x,y,z = getXYZ(df,args)
    r, ro, theta, phi = spherical(x,y,z)
    df['BeamCal_r'] = r
    df['BeamCal_ro'] = ro
    df['BeamCal_theta'] = theta
    df['BeamCal_phi'] = phi

    df.to_csv(fname+'_sphere_param'+'.csv',sep='|')

def rotate_abt_y(x,y,z,cross_angle):
    with np.nditer([x,y,z,None,None,None]) as it:
        for i,j,k,l,m,n in it:
            v = TVector3()
            v.SetXYZ(i,j,k)
            v.RotateY(cross_angle)
            l[...] = v.X()
            m[...] = v.Y()
            n[...] = v.Z()
        return (it.operands[3],it.operands[4],it.operands[5])

"""
#def add_rotated_param(df,cross_angle):
def add_rotated_param(inFile):
    fname = inFile.split('.')
    fname = fname[0]+'.'+fname[1]
    print(fname)
    #df = pd.read_csv(inFile,sep='|')
    df = pd.read_hdf(inFile,key='df')
    df = df.copy()
    args = ['mcp_endx', 'mcp_endy','mcp_endz','mcp_px','mcp_py','mcp_pz','mcp_e']
    df_zp = df.loc[df['mcp_endz']>0,args]
    df_zn = df.loc[df['mcp_endz']<0,args]
    dfs = [df_zp,df_zn]
    cross_angle = 7*10**(-3)
    cross_angles = [cross_angle, -cross_angle]

    for df, cross_angle in zip(dfs,cross_angles):
        args = ['mcp_endx', 'mcp_endy','mcp_endz']
        x,y,z = getXYZ(df,args)

        x_rot,y_rot,z_rot = rotate_abt_y(x,y,z,cross_angle)
        df.loc[:,'mcp_endx'] = x_rot
        df.loc[:,'mcp_endy'] = y_rot
        df.loc[:,'mcp_endz'] = z_rot
        r, ro, theta, phi = spherical(x_rot,y_rot,z_rot)
        df.loc[:,'mcp_endr'] = r
        df.loc[:,'mcp_endro'] = ro
        df.loc[:,'mcp_endtheta'] = theta
        df.loc[:,'mcp_endphi'] = phi

    f = [df_zp, df_zn]
    df_mcp = pd.concat(f,axis=0)

    df = pd.read_hdf(inFile,key='df')
    df = df.copy()
    #print(df.columns)
    args = ['BeamCal_x', 'BeamCal_y','BeamCal_z','BeamCal_e']
    df_zp = df.loc[df['BeamCal_z']>0,args]
    #print(df_zp.columns)
    df_zn = df.loc[df['BeamCal_z']<0,args]

    dfs = [df_zp,df_zn]
    cross_angles = [cross_angle, -cross_angle]

    for df, cross_angle in zip(dfs,cross_angles):
        args = ['BeamCal_x', 'BeamCal_y','BeamCal_z']
        x,y,z = getXYZ(df,args)

        x_rot,y_rot,z_rot = rotate_abt_y(x,y,z,cross_angle)
        df.loc[:,'BeamCal_x'] = x_rot
        df.loc[:,'BeamCal_y'] = y_rot
        df.loc[:,'BeamCal_z'] = z_rot

        r, ro, theta, phi = spherical(x_rot,y_rot,z_rot)
        df.loc[:,'BeamCal_r'] = r
        df.loc[:,'BeamCal_ro'] = ro
        df.loc[:,'BeamCal_theta'] = theta
        df.loc[:,'BeamCal_phi'] = phi
    f = [df_zp, df_zn]
    df_BeamCal = pd.concat(f,axis=0)

    df = [df_mcp,df_BeamCal]
    df = pd.concat(df,axis = 1)
    df.to_hdf(fname+'_rotated'+'.h5',key='df')
    #df.to_csv(fname+'_rotated_param'+'.csv',sep='|')
    #print(fname+'_rotated_param'+'.csv is created')
    print(fname+'_rotated'+'.h5 is created')


def bcal_rot(inFile, inFile_rot):
    #df = pd.read_csv(inFile,sep = '|')
    df = pd.read_hdf(inFile,key='df')
    fname = inFile.split('.')
    fname = fname[0]+'.'+fname[1]
    #df_rot = pd.read_csv(inFile_rot,sep='|')
    df_rot = pd.read_hdf(inFile_rot,key='df')
    dfs = [df,df_rot]
    args = ['BeamCal_x', 'BeamCal_y','BeamCal_z']
    bas = ['before','after']
    fig,axs = plt.subplots(nrows = 2, ncols = 2, sharex = False, sharey = True, tight_layout = True)
    plt.jet()
    for i,(df,ba) in enumerate(zip(dfs,bas)):
        clrs = ['green', 'red']
        titles = ['BeamCal_z>0','BeamCal_z<0']
        for j,(title,color) in enumerate(zip(titles,clrs)):
            temp = df.copy()
            temp = temp.query(title)
            print(title)
            print(temp['BeamCal_z'])
            x,y,_ = getXYZ(temp,args)
            _,_,_,pcm=axs[i,j].hist2d(x=x,y=y,range=None, bins =50,norm=colors.LogNorm(vmax=3000))
            fig.colorbar(pcm,ax=axs[i,j])
            #axs[i,j].scatter(x,y,s=1,facecolor = color)
            textstr = 'entries:' + str(x.shape[0])
            #set_axs(axs[i,j], title = ba+' rot '+title+'('+str(i)+','+str(j)+')', xlabel= 'x(mm)',ylabel ='y(mm)',textstr = textstr)
            set_axs(axs[i,j], title = ba+' rot '+title, xlabel= 'x(mm)',ylabel ='y(mm)',textstr = textstr)
    plt.savefig(fname+'_bcal.png')
    print(fname+'_bcal.png created')

def df_cut(df):
    bcal_zmax = df['BeamCal_z'].max()
    bcal_zmin = df['BeamCal_z'].min()
    bcal_romin = df['BeamCal_ro'].min()
    print('bcal_romin ',bcal_romin)
    bcal_romax = df['BeamCal_ro'].max()
    print('bcal_romax ',bcal_romax)
    df = df.query('mcp_endz<=@bcal_zmax & mcp_endz>=@bcal_zmin')
    df = df.query('mcp_endro<=@bcal_romax & mcp_endro>=@bcal_romin')

    #Initial Cut from BeamCal
    #df1 = df.query('mcp_endz<=@bcal_zmax & mcp_endz>=@bcal_zmin')
    #df = df.query('mcp_endro<=@bcal_romax & mcp_endro>=@bcal_romin')
    #args_mcp = ['mcp_endz', 'mcp_endro', 'mcp_endphi']
    #df = df.query(f'({args_mcp[0]}<= @bcal_zmax or {args_mcp[0]}>= @bcal_zmin) & {args_mcp[1]}<= @bcal_romax')
    return df


def bpipe_cut(df):
    path ='bpipe_cut_plots'
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)           #removes all the subdirectories!
        os.makedirs(path)
    #os.chdir(path)

    df = df.copy()
    df = df.astype(float)
    df = df_cut(df)
    #print(df['mcp_endphi'])

    phi_max = math.radians(156)
    phi_min = math.radians(-156)

    df_right = df.query('BeamCal_phi<=@phi_max & BeamCal_phi>=@phi_min')
    ro_right_min = df_right['BeamCal_ro'].min()
    print('ro_right_min ', ro_right_min)
    df_right = df_right.query('mcp_endro>=@ro_right_min')
    #histogram check
    args=['BeamCal_x','BeamCal_y','BeamCal_z']
    x,y,_=getXYZ(df_right,args)
    plt_hist2d(x,y,bins=50,title='BeamCal_right',xlabel='x(mm)',ylabel='y(mm)',fname=path+'/'+'bcal_right')

    df_left = df.query('BeamCal_phi>@phi_max or BeamCal_phi<@phi_min')
    ro_left_min = df_left['BeamCal_ro'].min()
    df_left = df_left.query('mcp_endro>=@ro_left_min')
    #Histogram check
    args=['BeamCal_x','BeamCal_y','BeamCal_z']
    x,y,_=getXYZ(df_left,args)
    plt_hist2d(x,y,bins=50,title='BeamCal_left',xlabel='x(mm)',ylabel='y(mm)',fname=path+'/'+'bcal_left')

    p = [df_left, df_right]
    df = pd.concat(p, axis = 0)
    x,y,_=getXYZ(df,args)
    plt_hist2d(x,y,bins=50,title='BeamCal_reco',xlabel='x(mm)',ylabel='y(mm)',fname=path+'/'+'bcal_reco')

    return (df, ro_right_min,ro_left_min)
    #return df
def bpipe_cut_mcp(df):
    path ='bpipe_cut_mcp_plots'
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)           #removes all the subdirectories!
        os.makedirs(path)
    #os.chdir(path)

    df = df.copy()
    df = df.astype(float)
    _,ro_right_min,ro_left_min = bpipe_cut(df)
    df = df_cut(df)

    #print(df['mcp_endphi'])

    phi_max = math.radians(156)
    phi_min = math.radians(-156)

    df_right = df.query('mcp_endphi<=@phi_max & mcp_endphi>=@phi_min')
    #ro_right_min = df_right['BeamCal_ro'].min()
    print('ro_right_min ', ro_right_min)
    df_right = df_right.query('mcp_endro>=@ro_right_min')
    #histogram check
    args=['mcp_endx','mcp_endy','mcp_endz']
    x,y,_=getXYZ(df_right,args)
    plt_hist2d(x,y,bins=50,title='mcp_end_right',xlabel='x(mm)',ylabel='y(mm)',fname=path+'/'+'mcp_right')

    df_left = df.query('mcp_endphi>@phi_max or mcp_endphi<@phi_min')
    #ro_left_min = df_left['BeamCal_ro'].min()
    df_left = df_left.query('mcp_endro>=@ro_left_min')
    #Histogram check
    args=['mcp_endx','mcp_endy','mcp_endz']
    x,y,_=getXYZ(df_left,args)
    plt_hist2d(x,y,bins=50,title='mcp_end_left',xlabel='x(mm)',ylabel='y(mm)',fname=path+'/'+'mcp_left')

    p = [df_left, df_right]
    df = pd.concat(p, axis = 0)
    x,y,_=getXYZ(df,args)
    plt_hist2d(x,y,bins=50,title='mcp_reco',xlabel='x(mm)',ylabel='y(mm)',fname=path+'/'+'mcp_reco')

    return df


def mcp_rot(inFile, inFile_rot):
    #df = pd.read_csv(inFile,sep = '|')
    df = pd.read_hdf(inFile,key='df')
    print('in mcp ',inFile)
    df = df_cut(df)
    fname = inFile.split('.')
    fname = fname[0]+'.'+fname[1]
    #df_rot = pd.read_csv(inFile_rot,key='df')
    df_rot = pd.read_hdf(inFile_rot,key='df')
    df_rot = df_cut(df_rot)
    dfs = [df,df_rot]
    args = ['mcp_endx', 'mcp_endy','mcp_endz']
    bas = ['before','after']
    fig,axs = plt.subplots(nrows = 2, ncols = 2, sharex = False, sharey = True, tight_layout = True)
    plt.jet()
    for i,(df,ba) in enumerate(zip(dfs,bas)):
        clrs = ['green', 'red']
        #titles = ['mcp_endz>0','mcp_endz<0']
        titles = ['mcp_endz>0','mcp_endz<0']
        for j,(title,color) in enumerate(zip(titles,clrs)):
            temp = df.copy()
            temp = temp.query(title)
            x,y,_ = getXYZ(temp,args)
            #axs[i,j].scatter(x,y,s=1,facecolor = color)
            _,_,_,pcm=axs[i,j].hist2d(x=x,y=y,range=None, bins =50,norm=colors.LogNorm(vmax=3000))
            fig.colorbar(pcm,ax=axs[i,j])
            textstr = 'entries:' + str(x.shape[0])
            #set_axs(axs[i,j], title = ba+' rot '+title+'('+str(i)+','+str(j)+')', xlabel= 'x(mm)',ylabel ='y(mm)',textstr = textstr)
            set_axs(axs[i,j], title = ba+' rot '+title, xlabel= 'x(mm)',ylabel ='y(mm)',textstr = textstr)
    plt.savefig(fname+'_mcp.png')
    print(fname+'_mcp.png created')



def hdf_from_root(inFilename):
    fname= inFilename.split('.')
    if(len(fname)==3):
        fname = fname[0]+'.'+fname[1]

    if ('/' in fname):
        fname = fname.split('/')[1]

    print(fname)
    tree = uproot.open(inFilename)['evtdata']
    df1 = tree.pandas.df(['mcp*'])
    print(df1.columns)
    #comment df2 for incoherent_pair
    df2 = tree.pandas.df(['BeamCal*'])
    f = [df1, df2]
    df = pd.concat(f,axis=1)
    outdir = 'hdf_file'
    df.to_hdf(outdir+fname+'.h5',key='df')
    return df

def add_sphere_param(inFile):
    fname= inFile.split('.')
    fname = fname[0]+'.'+fname[1]
    df = pd.read_hdf(inFile,key='df')
    df = df.copy()
    df = df.dropna()
    args = ['mcp_endx', 'mcp_endy','mcp_endz']
    x,y,z = getXYZ(df,args)
    r, ro, theta, phi = spherical(x,y,z)
    df.loc[:,'mcp_endr'] = r
    df.loc[:,'mcp_endro'] = ro
    df.loc[:,'mcp_endtheta'] = theta
    df.loc[:,'mcp_endphi'] = phi


    args = ['BeamCal_x', 'BeamCal_y','BeamCal_z']
    x,y,z = getXYZ(df,args)
    r, ro, theta, phi = spherical(x,y,z)
    df.loc[:,'BeamCal_r'] = r
    df.loc[:,'BeamCal_ro'] = ro
    df.loc[:,'BeamCal_theta'] = theta
    df.loc[:,'BeamCal_phi'] = phi

    df.to_hdf(fname+'.h5',key='df')
    #outdir = 'csv_files'
    #if not os.path.exists(outdir):
    #    os.makedirs(outdir)
    #df = pd.read_hdf(inFile,key='df')
    #df.to_csv(outdir+'/'+fname+'_add_phi_loc'+'.csv',sep='|')
    #df.to_root(outdir+'/'+fname+'_add_phi_loc'+'.root',key='evtdata')
    #df.to_root(fname+'.root',key='evtdata;1')
    outdir =''
    print(outdir+'/'+fname+'.h5 is created')


def plot_args(df,args,fig,axs,fname,directory,counter):
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

#inFile, fname = inFile()
#print(fname)
def auto_plot_from_hdf(inFilename):
    df = pd.read_hdf(inFilename, key='df')
    fname= inFilename.split('.')
    fname = fname[0]+'.'+fname[1]
    print(fname)
    directory = fname
    directory = directory+'_plots'
    if not os.path.exists(directory):
        os.makedirs(directory)
    print('directory ',directory)

    df2 = df.filter(regex='BeamCal_*')
    df3 = df.filter(regex='mcp_end*')
    df4 = df.filter(regex='mcp_start*')
    df5 = df[['mcp_e','mcp_px','mcp_py','mcp_pz']]

    df = [df2, df3, df4, df5]
    df = pd.concat(df,axis=1)

    args=[]
    n = 0
    m = 0
    for counter,col in enumerate(df.columns):
        args.append(col)
        print(counter)
        if (counter+1)%4==0 :
            fig, axs= plt.subplots(nrows=2,ncols=2,sharex=False, sharey=True,figsize=(15,18))
            plot_args(df,args,fig,axs,fname,directory,counter)
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



def mcp_rot_bpipe_cut(inFile, inFile_rot):
    #df = pd.read_csv(inFile,sep = '|')
    df = pd.read_hdf(inFile,key='df')
    print('in mcp ',inFile)
    df = bpipe_cut_mcp(df)
    fname = inFile.split('.')
    fname = fname[0]+'.'+fname[1]
    #df_rot = pd.read_csv(inFile_rot,key='df')
    df_rot = pd.read_hdf(inFile_rot,key='df')
    df_rot = bpipe_cut_mcp(df_rot)
    dfs = [df,df_rot]
    args = ['mcp_endx', 'mcp_endy','mcp_endz']
    bas = ['before','after']
    fig,axs = plt.subplots(nrows = 2, ncols = 2, sharex = False, sharey = True, tight_layout = True)
    plt.jet()
    for i,(df,ba) in enumerate(zip(dfs,bas)):
        clrs = ['green', 'red']
        #titles = ['mcp_endz>0','mcp_endz<0']
        titles = ['mcp_endz>0','mcp_endz<0']
        for j,(title,color) in enumerate(zip(titles,clrs)):
            temp = df.copy()
            temp = temp.query(title)
            x,y,_ = getXYZ(temp,args)
            #axs[i,j].scatter(x,y,s=1,facecolor = color)
            _,_,_,pcm=axs[i,j].hist2d(x=x,y=y,range=None, bins =50,norm=colors.LogNorm(vmax=3000))
            fig.colorbar(pcm,ax=axs[i,j])
            textstr = 'entries:' + str(x.shape[0])
            #set_axs(axs[i,j], title = ba+' rot '+title+'('+str(i)+','+str(j)+')', xlabel= 'x(mm)',ylabel ='y(mm)',textstr = textstr)
            set_axs(axs[i,j], title = ba+' rot '+title, xlabel= 'x(mm)',ylabel ='y(mm)',textstr = textstr)
    plt.savefig(fname+'_mcp_bpipe_cut.png')
    print(fname+'_mcp_bpipe_cut.png created')

