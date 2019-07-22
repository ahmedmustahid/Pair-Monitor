import numpy as np
import pandas as pd
from util_plot import spherical, getXYZ,inFile
import os



inFile, fname = inFile()
if ('/' in fname):
    fname = fname.split('/')[1]

df = pd.read_csv(inFile,sep='|')
def add_sphere_param(df,fname):
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
    outdir = 'csv_files'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    df.to_csv(outdir+'/'+fname+'_add_phi_loc'+'.csv',sep='|')
    print(outdir+'/'+fname+'_add_phi_loc'+'.csv is created')

add_sphere_param(df,fname)
