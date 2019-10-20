import os,shutil

def makeDirectory(dirname):
     if not os.path.exists(dirname):
         os.makedirs(dirname)


basedir="."

bases=os.listdir(basedir)

for base in bases:
    if os.path.isdir(base) and base.split('_')[0]=="Run":
        root = base+"/"+"images"
        if os.path.exists(root):
            shutil.rmtree(root)
            print root+" deleted"
            makeDirectory(root)
            print "empty "+root+" created"
