import os,shutil

basedir="."

bases=os.listdir(basedir)

for base in bases:
    if os.path.isdir(base) and base.split('_')[0]=="Run":
        root = base+"/"+"root"
        if os.path.exists(root):
            roots= os.listdir(root)
            print roots
            #for rt in roots:
            #    if os.path.isfile(rt):

