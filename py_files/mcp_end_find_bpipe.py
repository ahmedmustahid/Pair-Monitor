import numpy as np
import pandas as pd
from util_plot import inFile, DfToNp, set_axs,getXYZ, cal_phi, cal_r
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

def rot(x,z,cross_angle):
    x1 = np.add( x * np.cos(cross_angle) , z * np.sin( cross_angle) )
    z1 = np.add( z * np.cos(cross_angle) , -1 * x * np.sin(cross_angle) )
    return (x1,z1)

#args_mcp = ['mcp_endr', 'mcp_endphi']
#args_bcal = ['BeamCal_z','BeamCal_r']
# => must be used in this order
#cut must be done after rotation
def df_cut(df,args_mcp):
    temp = df[args_mcp[1]]
    temp = DfToNp(temp)
    temp = np.rad2deg(temp)
    df[args_mcp[1]] = temp
    print(df[args_mcp[1]])
    df_left = df.query(f'( {args_mcp[1]}<-145 or {args_mcp[1]}>145 ) & {args_mcp[0]}>81.19')
    df_right = df.query(f'( {args_mcp[1]}<-145 or {args_mcp[1]}>145 ) & {args_mcp[0]}>25.45 ')
    p = [df_right, df_left]
    df = pd.concat(p, axis = 0)

    return df




inFile, fname = inFile()

df = pd.read_csv(inFile,sep ='|')



args = ['mcp_endx', 'mcp_endy','mcp_endz']
args_bcal = ['BeamCal_z', 'BeamCal_r', 'BeamCal_phi']

bcal_zmax = df[args_bcal[0]].max()
print('bcal_zmax ',bcal_zmax)
bcal_zmin = df[args_bcal[0]].min()
print('bcal_zmin',bcal_zmin)
bcal_rmax = df[args_bcal[1]].max()
print('bcal_rmax',bcal_rmax)

args_mcp = ['mcp_endz', 'mcp_endr', 'mcp_endphi']
#Initial Cut from BeamCal
df = df.query(f'({args_mcp[0]}<= @bcal_zmax or {args_mcp[0]}>= @bcal_zmin) & {args_mcp[1]}<= @bcal_rmax')
print(df[['mcp_endr','mcp_endx']])
df1 = df.loc[df['mcp_endz']>0, args ]
#df1 = df.query('mcp_endz>0')
print(df1['mcp_endz'])
x,y,z = getXYZ(df1,args)

cross_angle = -7*10**(-3)

#Rotation
x1, z1 = rot(x,z,cross_angle)

x2 = x1 - x

print('df1')
df1['mcp_endx1'] = x1
df1['mcp_endz1'] = z1
df1['mcp_endx2']= x2


print('df1')
print(df1['mcp_endx2'])

print('df1 x max')
print(df1['mcp_endx'].min())

print('df1 x1 max')
print(df1['mcp_endx1'].min())
r1 = np.hypot(x1,y)
phi1 = cal_phi(x1,y)
df1['mcp_endr1'] = r1
df1['mcp_endphi1'] = phi1

#Positive z; cut after rotation
#args now correspond to the rotated variables
args_mcp = ['mcp_endr1', 'mcp_endphi1']
#df1 = df_cut(df1, args_mcp)





#Negative z
#df2 = df.query('mcp_endzs<>0')
df2 = df.loc[df['mcp_endz']<0, args ]
x,y,z = getXYZ(df2,args)

cross_angle = -7*10**(-3)

x1, z1 = rot(x,z,cross_angle)

x2 = x1 - x

df2['mcp_endx1'] = x1
df2['mcp_endz1'] = z1
df2['mcp_endx2']= x2
print('df2')
print(df2['mcp_endx2'])

r1 = np.hypot(x1,y)
df2['mcp_endr1'] = r1
df2['mcp_endphi1']= cal_phi(x1,y)
#cut after rotation
#args now correspond to the rotated variables
args_mcp = ['mcp_endr1', 'mcp_endphi1']
#df2 = df_cut(df2, args_mcp)




arg1 = ['mcp_endx', 'mcp_endy','mcp_endz']
arg2 = ['mcp_endx1', 'mcp_endy','mcp_endz1']
args = [arg1 , arg2]
bas = ['before','after']
#fig,axs = plt.subplots(nrows = 2, ncols = 2, sharex = False, sharey = True, tight_layout = True,figsize =(10,18))
fig,axs = plt.subplots(nrows = 2, ncols = 2, sharex = False, sharey = True, tight_layout = True)
plt.jet()
for i,(arg,ba) in enumerate(zip(args,bas)):
    dfs = [df1, df2]
    clrs = ['green', 'red']
    titles = ['z>0','z<0']
    #axs = axs.flatten()
    for j,(title,df,color) in enumerate(zip(titles,dfs,clrs)):
        x,y,_ = getXYZ(df,arg)
        #axs[i,j].scatter(x,y,s=1,facecolor = color)
        _,_,_,pcm=axs[i,j].hist2d(x=x,y=y,range=None, bins =50,norm=colors.LogNorm())
        fig.colorbar(pcm,ax=axs[i,j])
        textstr = 'entries:' + str(x.shape[0])
        set_axs(axs[i,j], title = ba+' rot '+title+'('+str(j)+','+str(i)+')', xlabel= 'x(mm)',ylabel ='y(mm)',textstr = textstr)
        #set_axs(axs[i,j], title = ba+' rot '+title, xlabel= 'x(mm)',ylabel ='y(mm)',textstr = textstr)


outdir = fname+'_plots'
if not os.path.exists(outdir):
    os.makedirs(outdir)

plt.savefig(outdir+'/'+fname+'_mcp_end1.png')
print(outdir+'/'+fname+'_mcp_end1.png created')


