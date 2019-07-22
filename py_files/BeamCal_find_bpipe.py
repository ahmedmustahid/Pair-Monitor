import numpy as np
import pandas as pd
from util_plot import inFile, DfToNp, set_axs,getXYZ
import matplotlib.pyplot as plt
import os

def rot(x,z,cross_angle):
    x1 = np.add( x * np.cos(cross_angle) , z * np.sin( cross_angle) )
    z1 = np.add( z * np.cos(cross_angle) , -1 * x * np.sin(cross_angle) )
    return (x1,z1)


inFile, fname = inFile()


df = pd.read_csv(inFile,sep ='|')

args = ['BeamCal_x', 'BeamCal_y','BeamCal_z']

df1 = df.loc[df['BeamCal_z']>0, args ]

x,y,z = getXYZ(df1,args)

cross_angle = -7*10**(-3)

x1, z1 = rot(x,z,cross_angle)

x2 = x1 - x

df1['BeamCal_x1'] = x1
df1['BeamCal_z1'] = z1
df1['x2']= x2

print('df1')
print(df1['x2'])


df2 = df.loc[df['BeamCal_z']<0, args ]
x,y,z = getXYZ(df2,args)

cross_angle = 7*10**(-3)

x1, z1 = rot(x,z,cross_angle)

x2 = x1 - x

df2['BeamCal_x1'] = x1
df2['BeamCal_z1'] = z1
df2['x2']= x2

print('df2')
print(df2['x2'])

arg1 = ['BeamCal_x', 'BeamCal_y','BeamCal_z']
arg2 = ['BeamCal_x1', 'BeamCal_y','BeamCal_z1']
args = [arg1 , arg2]
bas = ['before','after']
#fig,axs = plt.subplots(nrows = 2, ncols = 2, sharex = False, sharey = True, tight_layout = True,figsize =(10,18))
fig,axs = plt.subplots(nrows = 2, ncols = 2, sharex = False, sharey = True, tight_layout = True)
for i,(arg,ba) in enumerate(zip(args,bas)):
    dfs = [df1, df2]
    colors = ['green', 'red']
    titles = ['z>0','z<0']
    #axs = axs.flatten()
    for j,(title,df,color) in enumerate(zip(titles,dfs,colors)):
        x,y,_ = getXYZ(df,arg)
        axs[i,j].scatter(x,y,s=1,facecolor = color)
        textstr = 'entries:' + str(x.shape[0])
        set_axs(axs[i,j], title = ba+' rot '+title+'('+str(j)+','+str(i)+')', xlabel= 'x(mm)',ylabel ='y(mm)',textstr = textstr)
        set_axs(axs[i,j], title = ba+' rot '+title, xlabel= 'x(mm)',ylabel ='y(mm)',textstr = textstr)
outdir = fname+'_plots'
if not os.path.exists(outdir):
    os.makedirs(outdir)
plt.savefig(outdir+'/'+fname+'_bcal1.png')
print(outdir+'/'+fname+'_bcal1.png created')


