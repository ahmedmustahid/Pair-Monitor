import os,string,shutil,sys,re,utils,conf

def getSubFilePath(absPath,subfilename):
#for subfilename in subfilenames:
    prefix_subfile = subfilename.split('_')[1]
    base_subfile = subfilename.split('/')[-1]
    inFile = absPath+"/"+prefix_subfile+"/"+base_subfile+".slcio"

    return inFile

def getxml(fpath,nxmlfile):
    ofname = ofname = conf.XMLDIR + "/"+conf.OUTPUT_PREFIX+"_" + str(nxmlfile) + ".xml"
    foutxml = open(ofname,'w')
    template = open(conf.TEMPLATEDIR+"/"+conf.TEMPLATEFILE,'r')

    for t in template:
        if re.search("__INPUTFILES__",t):
            foutxml.write(re.sub("__INPUTFILES__",fpath,t))
        elif re.search("__OUTPUTROOTDIR__/__OUTPUTROOT__",t):
            tmp = re.sub("__OUTPUTROOTDIR__",conf.OUTDIR_ROOT,t)
            tmp2 = conf.SPACE + re.sub("__OUTPUTROOT__",conf.OUTPUT_PREFIX+ "_" + str(nxmlfile) + ".root",tmp)
            foutxml.write(tmp2)
        elif re.search("__GEARFILE__",t):
            tmp = conf.SPACE + re.sub("__GEARFILE__",conf.GEARFILE,t)
            foutxml.write(tmp)
        elif re.search("__INPUTPREFIX__",t):
            tmp = conf.SPACE + re.sub("__INPUTPREFIX__",conf.INPUT_PREFIX,t)
            foutxml.write(tmp)
        else:
            foutxml.write(t)

def writeXMLs(absPath, nFilesPerShot):
    filenames = getFileName()

    for subfilenames in filenames:
        nxmlfile=0
        while(len(subfilenames)):

            #nxmlfile = 0
            nxmlfile+=1

            len_subfiles = len(subfilenames)
            if(len_subfiles<nFilesPerShot):
                for i in range(len_subfiles):
                    if not i==len_subfiles-1:
                        fpath+=getSubfilename(absPath, subfilenames.pop())+'\n'
                    else:
                        fpath+=getSubfilename(absPath, subfilenames.pop())
                getxml(fpath,nxmlfile)


            for i in nFilesPerShot:
                if not i==nFilesPerShot-1:
                    fpath+=getSubfilename(absPath, subfilenames.pop())+'\n'
                else:
                    fpath+=getSubfilename(absPath, subfilenames.pop())
                getxml(fpath,nxmlfile)

utils.makeDirectory(conf.TEMPLATEDIR)
utils.makeDirectory(conf.XMLDIR)
utils.makeDirectory(conf.OUTDIR_ROOT)

shutil.copy(conf.TEMPLATEXMLORIGDIR+"/"+conf.TEMPLATEFILE,conf.TEMPLATEDIR+"/"+conf.TEMPLATEFILE)

absPath = "/home/belle2/mustahid/CAINPlus/CAINPlus/run/datasets/all"
nFilesPerShot= conf.nFilesPerShot
writeXMLs(absPath,nFilesPerShot)


