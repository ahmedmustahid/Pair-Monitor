import os,shutil

basedir="."

bases=os.listdir(basedir)

for base in bases:
    if os.path.isdir(base) and base.split('_')[0]=="Run":
        root = base+"/"+"generatedXMLs"
        if os.path.exists(root):
            shutil.rmtree(root)
            print root
