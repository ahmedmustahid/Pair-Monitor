import numpy as np
import pandas as pd
from util_plot import spherical, getXYZ,inFile
import root_pandas



inFile, fname = inFile()
df = pd.read_csv(inFile,sep='|')
def add_sphere_param(df,fname):
    df = df.copy()
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

    df.to_root(fname+'_sphere_param'+'.root',key='evtdata')
    #df.to_csv(fname+'_sphere_param'+'.csv',sep='|')

add_sphere_param(df,fname)
