
import os, shutil
from pathlib import Path

#choice of Sigx or Sigy is based on the position of split

imdirs = Path.cwd()/'images'
xdirs=[]
for imdir in imdirs.iterdir():
    if "_" in imdir.name:
        imdir_split=imdir.name.split('_')
        imdirfin= imdir_split[1]
        xdirs.append(imdirfin)

xdirs=set(xdirs)

for xdir in xdirs:
    xdirpath = imdirs/xdir
    if not xdirpath.exists():
        xdirpath.mkdir(parents=True, exist_ok=True)
        print(xdir)

for imdir in imdirs.iterdir():
    if "_" in imdir.name:
        imdir_split=imdir.name.split('_')[1]
        if imdir_split in xdirs:
            #print(imdir_split)
            #print(imdir_split)
            for image in imdir.rglob("*.png"):
                target = str(imdirs)+"/"+ imdir_split
                #print("source ",image)
                #print("target ", target)
                #break
                shutil.move(str(image),target)
            print("moved from Run_",imdir_split)
            shutil.rmtree(str(imdir))

imdirs.rename("X_images")
print("Renamed images/ to X_images/")
