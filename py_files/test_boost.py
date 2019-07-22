import numpy as np
import pandas as pd
from util_plot import inFile,boost, plt_hist2d, set_axs, momentum_aftr_boost,getXYZ,spherical,rotate_abt_y
import matplotlib.pyplot as plt


inFile,fname = inFile()

df = pd.read_csv(inFile,sep='|')

"""
args = ['mcp_px','mcp_py','mcp_pz','mcp_e']
px,_,_,_ = momentum_aftr_boost(df,args)
#with np.errstate(invalid='ignore'):
#y = np.isfinite(px)

#with np.nditer([y]) as it:
#    for i in it:
#        if i== False:
#            print('0')
#
fig, ax= plt.subplots(nrows=1,ncols= 1 ,sharex=False, sharey=True,figsize=(5,7))
textstr = 'entries: '+str(px.shape[0])
set_axs(ax,title=r'$P_x$ after boost only $\beta_x$',xlabel=r'$P_x$(GeV)',textstr=textstr)
ax.hist(px,bins=50,log=True,histtype='step')
plt.savefig('px_aftr_boost2.png')
#plt.show()
#args = ['BeamCal_px','BeamCal_py','BeamCal_pz','BeamCal_e']
"""

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
print(df.columns)
args = ['BeamCal_x', 'BeamCal_y','BeamCal_z','BeamCal_e']
df_zp = df.loc[df['BeamCal_z']>0,args]
#print(df_zp.coulmns)
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

