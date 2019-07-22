import numpy as np
import pandas as pd
from util_plot import inFile,getXYZ, spherical
import matplotlib.pyplot as plt

inFile, fname = inFile()
df = pd.read_csv(inFile,sep='|')

args = ['BeamCal_x', 'BeamCal_y','BeamCal_z']

x,y,z = getXYZ(df,args)
r, ro,theta,phi = spherical(x,y,z)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(phi,bins=100, range=None, histtype='bar',log='True')
plt.show()
