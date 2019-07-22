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
    fname = inFile.split('.')[0]

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

#def add_rotated_param(df,cross_angle):
def add_rotated_param(inFile):
    fname = inFile.split('.')[0]
    df = pd.read_csv(inFile,sep='|')
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

    df = pd.read_csv(inFile,sep='|')
    df = df.copy()
    print(df.columns)
    args = ['BeamCal_x', 'BeamCal_y','BeamCal_z','BeamCal_e']
    df_zp = df.loc[df['BeamCal_z']>0,args]
    print(df_zp.columns)
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

    df.to_csv(fname+'_rotated_param'+'.csv',sep='|')
    print(fname+'_rotated_param'+'.csv is created')


def bcal_rot(inFile, inFile_rot):
    df = pd.read_csv(inFile,sep = '|')
    fname = inFile.split('.')[0]
    df_rot = pd.read_csv(inFile_rot,sep='|')
    dfs = [df,df_rot]
    args = ['BeamCal_x', 'BeamCal_y','BeamCal_z']
    bas = ['before','after']
    fig,axs = plt.subplots(nrows = 2, ncols = 2, sharex = False, sharey = True, tight_layout = True)
    for i,(df,ba) in enumerate(zip(dfs,bas)):
        colors = ['green', 'red']
        titles = ['BeamCal_z>0','BeamCal_z<0']
        for j,(title,color) in enumerate(zip(titles,colors)):
            temp = df.copy()
            temp = temp.query(title)
            print(title)
            print(temp['BeamCal_z'])
            x,y,_ = getXYZ(temp,args)
            axs[i,j].scatter(x,y,s=1,facecolor = color)
            textstr = 'entries:' + str(x.shape[0])
            #set_axs(axs[i,j], title = ba+' rot '+title+'('+str(i)+','+str(j)+')', xlabel= 'x(mm)',ylabel ='y(mm)',textstr = textstr)
            set_axs(axs[i,j], title = ba+' rot '+title, xlabel= 'x(mm)',ylabel ='y(mm)',textstr = textstr)
    plt.savefig(fname+'_bcal1.png')
    print(fname+'_bcal1.png created')

def df_cut(df):
    args = ['mcp_endx', 'mcp_endy','mcp_endz']
    args_bcal = ['BeamCal_z', 'BeamCal_ro', 'BeamCal_phi']
    bcal_zmax = df[args_bcal[0]].max()
    print('bcal_zmax ',bcal_zmax)
    bcal_zmin = df[args_bcal[0]].min()
    print('bcal_zmin',bcal_zmin)
    bcal_romax = df[args_bcal[1]].max()
    print('bcal_romax',bcal_romax)
    args_mcp = ['mcp_endz', 'mcp_endro', 'mcp_endphi']
    #Initial Cut from BeamCal
    df = df.query(f'({args_mcp[0]}<= @bcal_zmax or {args_mcp[0]}>= @bcal_zmin) & {args_mcp[1]}<= @bcal_romax')
    return df

def mcp_rot(inFile, inFile_rot):
    df = pd.read_csv(inFile,sep = '|')
    df = df_cut(df)
    fname = inFile.split('.')[0]
    df_rot = pd.read_csv(inFile_rot,sep='|')
    df_rot = df_cut(df_rot)
    dfs = [df,df_rot]
    args = ['mcp_endx', 'mcp_endy','mcp_endz']
    bas = ['before','after']
    fig,axs = plt.subplots(nrows = 2, ncols = 2, sharex = False, sharey = True, tight_layout = True)
    plt.jet()
    for i,(df,ba) in enumerate(zip(dfs,bas)):
        clrs = ['green', 'red']
        #titles = ['mcp_endz>0','mcp_endz<0']
        titles = ['BeamCal_z>0','BeamCal_z<0']
        for j,(title,color) in enumerate(zip(titles,clrs)):
            temp = df.copy()
            temp = temp.query(title)
            x,y,_ = getXYZ(temp,args)
            #axs[i,j].scatter(x,y,s=1,facecolor = color)
            _,_,_,pcm=axs[i,j].hist2d(x=x,y=y,range=None, bins =50,norm=colors.LogNorm())
            fig.colorbar(pcm,ax=axs[i,j])
            textstr = 'entries:' + str(x.shape[0])
            #set_axs(axs[i,j], title = ba+' rot '+title+'('+str(i)+','+str(j)+')', xlabel= 'x(mm)',ylabel ='y(mm)',textstr = textstr)
            set_axs(axs[i,j], title = ba+' rot '+title, xlabel= 'x(mm)',ylabel ='y(mm)',textstr = textstr)
    plt.savefig(fname+'_mcp.png')
    print(fname+'_mcp.png created')






