#!/usr/bin/python
import os,re,glob,conf,shutil
#from random import sample

#def makeDirectory(dirname):
#        if not os.path.exists(dirname):
#                os.mkdir(dirname)
def makeDirectory(dirname):
     if not os.path.exists(dirname):
         os.makedirs(dirname)

def removeMetaCharacter(strin):
	strout = re.sub("\*","",strin)
	strout = re.sub("\,","",strout)
	strout = re.sub("\ ","",strout)
	return strout



class SetParameterValues():
    def __init__(self,param,values):
        self.param = param
        self.values = values

    def getParameterWithValues(Self):
        retvals = []
        for value in self.values:
            retvals.append(self.param+value)
        print 'retvals', retvals
        return retvals
#class SetParameterValues_confy():
#	def __init__(self,param,values):
#		self.param = param
#		self.values = values
#	def getParamWithValues(self):
#		retvals = []
#		for value in self.values:
#			retvals.append(self.param+value)
#		return retvals


class RunParameterSet():
	def __init__(self):
		self.params = []
		self.values = []
		self.startevts = []
	def addParameterSet(self,param,value):
		if param not in self.params:
			self.params.append(param)
			self.values.append(value)

	def addStartEvent(self,evt):
		self.startevts.append(evt)

	def getRunID(self):
		runid = ""
		for i in range(len(self.params)):
			runid += "_" + removeMetaCharacter(self.params[i]) + removeMetaCharacter(self.values[i])
		return runid
	def getDirName(self):
		dirname = "/CainFiles_Sigx/Run" + self.getRunID()
		#dirname = "./Run" + self.getRunID()
        #dirname = dirname + str(sample())
		return dirname
	def getDirName_grid(self):
		dirname = "./Run" + self.getRunID()
        #dirname = dirname + str(sample())
		return dirname



#def getRunInfoList():
#	runinfolist = []
#	for scanparameter in conf.scandata:
#		if len(runinfolist) == 0:
#			for value in scanparameter.values:
#				newinfo = RunParameterSet()
#				newinfo.addParameterSet(scanparameter.param,value)
#				nsubmit = conf.nTotalEvents/conf.nEventPerSubmit + 1
#                for isub in range(nsubmit):
#					newinfo.addStartEvent(isub*conf.nEventPerSubmit)
#				if conf.nTotalEvents > isub*conf.nEventPerSubmit:
#					newinfo.addStartEvent(conf.nTotalEvents) # last event
#				runinfolist.append(newinfo)
#		else :
#			newruninfolist = []
#			for currentinfo in runinfolist:
#				for value in scanparameter.values:
#					newinfo = RunParameterSet()
#					#duplicate existing info
#					for i in range(len(currentinfo.params)):
#						newinfo.addParameterSet(currentinfo.params[i],currentinfo.values[i])
#					#add new info
#					newinfo.addParameterSet(scanparameter.param,value)
#					newruninfolist.append(newinfo)
#			runinfolist = newruninfolist
#
#	return runinfolist


def getRunInfoList():
    runinfolist=[]
    #print 'len scandata ', len(conf.scandata)
    #print 'conf scandata ',conf.scandata
    #for scandata in conf.scandata:
    #    print 'scan param ', scandata.param
    #    print 'val scandt ', scandata.values
    for scanparameter in conf.scandata:
        #print scanparameter.values
        if len(runinfolist)==0:
            for value in scanparameter.values:
                newinfo = RunParameterSet()
                newinfo.addParameterSet(scanparameter.param,value)
                #nsubmit = conf.nTotalEvents/conf.nEventPerSubmit + 1
                nsubmit = conf.nTotalEvents/conf.nEventPerSubmit
                #print('nsubmit ', nsubmit)
                for isub in range(nsubmit):
                    #print 'isub ',isub
                    #print "new start nevent = " + str(isub*conf.nEventPerSubmit)
                    newinfo.addStartEvent(isub*conf.nEventPerSubmit)
                if conf.nTotalEvents > isub* conf.nEventPerSubmit:
                    newinfo.addStartEvent(conf.nTotalEvents)
                runinfolist.append(newinfo)
        else:
            newruninfolist=[]
            for currentinfo in runinfolist:
                for value in scanparameter.values:
                    newinfo = RunParameterSet()
                    for i in range(len(currentinfo.params)):
                        newinfo.addParameterSet(currentinfo.params[i],currentinfo.values[i])
                    newinfo.addParameterSet(scanparameter.param, value)
                    for i in currentinfo.startevts:
                      #print "currentinfo.startevts = " + str(i)
                      newinfo.addStartEvent(i)
                    newruninfolist.append(newinfo)

            runinfolist= newruninfolist
            #print len(runinfolist)
            #for i in runinfolist:
            #  print i.params[0], i.values[0], i.params[1], i.values[1]
    return runinfolist


#def getRunInfoList_confy():
#	runinfolist = []
#	for scanparameter in confy.scandata:
#		if len(runinfolist) == 0:
#			for value in scanparameter.values:
#				newinfo = RunParameterSet()
#				newinfo.addParameterSet(scanparameter.param,value)
#				nsubmit = confy.nTotalEvents/confy.nEventPerSubmit + 1
#				for isub in range(nsubmit):
#					newinfo.addStartEvent(isub*confy.nEventPerSubmit)
#				if confy.nTotalEvents > isub*confy.nEventPerSubmit:
#					newinfo.addStartEvent(confy.nTotalEvents) # last event
#				runinfolist.append(newinfo)
#		else :
#			newruninfolist = []
#			for currentinfo in runinfolist:
#				for value in scanparameter.values:
#					newinfo = RunParameterSet()
#					#duplicate existing info
#					for i in range(len(currentinfo.params)):
#						newinfo.addParameterSet(currentinfo.params[i],currentinfo.values[i])
#					#add new info
#					newinfo.addParameterSet(scanparameter.param,value)
#					newruninfolist.append(newinfo)
#			runinfolist = newruninfolist
#
#	return runinfolist



#def getRunInfoList_confy():
#	runinfolist = []
#	for scanparameter in confy.scandata_confy:
#		if len(runinfolist) == 0:
#			for value in scanparameter.values:
#				newinfo = RunParameterSet()
#				newinfo.addParameterSet(scanparameter.param,value)
#				nsubmit = confy.nTotalEvents/confy.nEventPerSubmit + 1
#				for isub in range(nsubmit):
#					newinfo.addStartEvent(isub*confy.nEventPerSubmit)
#				if confy.nTotalEvents > isub*confy.nEventPerSubmit:
#					newinfo.addStartEvent(confy.nTotalEvents) # last event
#				runinfolist.append(newinfo)
#		else :
#			newruninfolist = []
#			for currentinfo in runinfolist:
#				for value in scanparameter.values:
#					newinfo = RunParameterSet()
#					#duplicate existing info
#					for i in range(len(currentinfo.params)):
#						newinfo.addParameterSet(currentinfo.params[i],currentinfo.values[i])
#					#add new info
#					newinfo.addParameterSet(scanparameter.param,value)
#					newruninfolist.append(newinfo)
#			runinfolist = newruninfolist
#
#	return runinfolist



#def getFileName():
#    #dirlist = []
#    filenames = []
#    runinfolist = getRunInfoList()
#    for runinfo in runinfolist:
#        subfilenames = []
#        #subdirlist = []
#        dirname = runinfo.getDirName()
#        #dirlist.append(dirlist)
#        for startevt in range(len(runinfo.startevts)-1):
#            subdirname = "ev_" + str(runinfo.startevts[startevt]) + "-" + str(runinfo.startevts[startevt+1])
#            subfilenames.append(dirname+"_"+subdirname)
#        filenames.append(subfilenames)
#            #subdirlist.append(subdirname)
#            #files = os.listdir(dirname+"/"+subdirname)
#            #infiles = []
#            #for afile in files:
#            #    suffix = '.' + afile.split('.')[-1]
#            #    if suffix == ".slcio":
#            #        absFilePath = os.path.abspath(afile)
#            #        infiles.append(absFilePath)
#    return filenames

#def oddrand():

def getFileNameList():
    #Define input directory
    runinfolist = getRunInfoList()

    FileNameList = []
    for runinfo in runinfolist:

        dirname = runinfo.getDirName()

        for startevt in range(len(runinfo.startevts)-1):
            #print 'startevt'+' '+str(startevt)
            subdirname = 'ev_' + str(runinfo.startevts[startevt]) + '-' + str(runinfo.startevts[startevt+1])
            filenamepath = dirname+'/'+subdirname+'/'
            #print(filenamepath)
            #filename= [os.path.abspath(x) for x in os.listdir(filenamepath)]
            #filename = os.path.join(filenamepath,filename)
            filenames = [os.path.abspath(x) for x in glob.iglob(filenamepath+'*.i')]
            #print(filenames)
            for filename in filenames:
                FileNameList.append(filename)
    #print(FileNameList)
    return FileNameList #List  of all files in a directory; not list of list, because all files are added from the lowest level

def ret_name_grid(infile):
    infile = infile.split('.')
    infile = infile[0]+'.'+infile[1]+'.'+infile[2]

    return infile


def writeXMLs(outfiledir,infilenames,nFilesPerShot):
    nxmlfile = 0
    rootdir = outfiledir+'/'+conf.OUTDIR_ROOT
    makeDirectory(rootdir)
    xmldir = outfiledir+'/'+conf.XMLDIR
    makeDirectory(xmldir)
    while(1):
        #print('infilenames len '+str(len(infilenames))+' nxmlfile '+str(nxmlfile))
        if (len(infilenames)<=0):
            break
        else:
            nxmlfile+=1
            slcioname = infilenames[len(infilenames)-1]
            slcioname = slcioname.split('/')[-1]
            fname = ret_name_grid(slcioname)
            #xmlname = xmldir+'/'+ fname +'_'+str(nxmlfile)+ '.xml'
            xmlname = xmldir+'/'+ fname + '.xml'
            #print(xmlname)
            if not os.path.exists(xmlname):
                print "writing "+xmlname.split('/')[-1]
                foutxml = open(xmlname,'w')
                template = open(conf.TEMPLATEDIR+'/'+conf.TEMPLATEFILE,'r')
                for t in template:
                    if re.search('__INPUTFILES__',t):
                        for i in range(conf.nFilesPerShot):
                            if(conf.nFilesPerShot>1 and i==0):
                                fpath = conf.SPACE + infilenames.pop()+'\n'
                            elif (conf.nFilesPerShot==1):
                                fpath = conf.SPACE + infilenames.pop()
                            elif (i == conf.nFilesPerShot-1):
                                fpath+= conf.SPACE + infilenames.pop()
                            else:
                                if(len(infilenames)==1):
                                    fpath+=conf.SPACE+infilenames.pop()
                                else:
                                    fpath +=conf.SPACE+infilenames.pop()+'\n'
                        foutxml.write(re.sub('__INPUTFILES__',fpath,t))
                    elif re.search('__OUTPUTROOTDIR__/__OUTPUTROOT__',t):
                        tmp = re.sub('__OUTPUTROOTDIR__',rootdir, t)
                        tmp2 = conf.SPACE+re.sub('__OUTPUTROOT__',fname+'.root',tmp)
                        #tmp2 = conf.SPACE+re.sub('__OUTPUTROOT__',fname+'_'+str(nxmlfile)+'.root',tmp)
                        foutxml.write(tmp2)
                    elif re.search('__GEARFILE__',t):
                        tmp = conf.SPACE + re.sub('__GEARFILE__',conf.GEARFILE,t)
                        foutxml.write(tmp)
                    elif re.search('__INPUTPREFIX__',t):
                        tmp = conf.SPACE + re.sub('__INPUTPREFIX__',conf.INPUT_PREFIX,t)
                        foutxml.write(tmp)
                    else:
                        foutxml.write(t)

            else:
                infilenames.pop()


def makeSteer_grid():
    #Define input directory
    runinfolist = getRunInfoList()


    #FileNameList = []
    for runinfo in runinfolist:

        dirname = conf.LISTDIR
        dirname = dirname+'/'+runinfo.getDirName_grid()
        gridsubdirs = os.listdir(dirname)
        outfiledir = runinfo.getDirName_grid()+'/'
        outfiledir = os.path.abspath(outfiledir)
        #print outfiledir
        #break
        for gridsubdir in gridsubdirs:
            filenamepath = dirname+'/'+gridsubdir+'/'
            infilenames = [os.path.abspath(x) for x in glob.iglob(filenamepath+'*.slcio')]
            if (len(infilenames)==0):
                break
            #for filename in infilenames:
            #    print(filename)
            writeXMLs(outfiledir,infilenames,conf.nFilesPerShot)

def manyRun_root():
    runinfolist = getRunInfoList()
    for runinfo in runinfolist:
        dirname = runinfo.getDirName_grid()
        xmldir = dirname +'/'+conf.XMLDIR
        xmls = [os.path.abspath(x) for x in glob.iglob(xmldir+'/'+'*.xml')]
        logdir = dirname + '/'+'xml_to_root_log'
        makeDirectory(logdir)
        rootdir = dirname+'/'+conf.OUTDIR_ROOT
        makeDirectory(rootdir)
        for xml in xmls:
            name = xml.split('/')[-1]
            name = ret_name_grid(name)
            cmd = 'bsub -q l '+'-o '+logdir+'/'+name+'.log '+'\"Marlin '+ xml +'\"'
            #print cmd
            os.system(cmd)

def manyRun_image():
    basedir="."

    shcmdlist =[]
    bases=os.listdir(basedir)
    for base in bases:
        if os.path.isdir(base) and base.split('_')[0]=="Run":
            root = base+"/"+"root"
            if os.path.exists(root):
                roots= os.listdir(root)
                for rootfile in roots:
                    #print nshfile
                    rootfiledir=root+"/"+rootfile
                    rootfiledir = os.path.abspath(rootfiledir)
                    pyfilename="rophi2.py"
                    pcmd="python "+ pyfilename +" "+rootfiledir+"\n"
                    shcmdlist.append(pcmd)
    fname=basedir+'/sh_dir'
    makeDirectory(fname)

    if os.path.exists(fname+"/my.sh"):
        open(fname+"/my.sh","w").close()
    i = 0
    print "len shcmdlist ",len(shcmdlist)

    global nshfile
    nshfile=0
    while(len(shcmdlist)):
        if(len(shcmdlist)<=0):
            break
        nshfile+=1
        if len(shcmdlist)>=250 and nshfile%250==0:
            fout=open(fname+"/my_"+str(nshfile)+".sh","w+")
            tmplist=[]
            for i in range(250):
                tmplist.append(shcmdlist.pop())
            fout.write("".join(tmplist))
            cmd = "bsub -q l "+"-o "+"rootfile.log "+'\"sh '+ fname+"/my_"+str(nshfile)+".sh"+'\"'
            fout.close()
            os.system(cmd)
            print cmd
            #break
        elif len(shcmdlist)<250:
            fout=open(fname+"/my_"+str(nshfile)+".sh","w+")
            tmplist=[]
            for i in range(len(shcmdlist)):
                tmplist.append(shcmdlist.pop())
            fout.write("".join(tmplist))
            cmd = "bsub -q l "+"-o "+"rootfile.log "+'\"sh '+ fname+"/my_"+str(nshfile)+".sh"+'\"'
            fout.close()
            os.system(cmd)
            print cmd
    print("nshfile ",nshfile)










    #while(1):

    #    nshfile+=1
    #    if(len(shcmdlist)<=0):
    #        break
    #    elif len(shcmdlist)>=250:
    #        #if os.path.exists(fname+"/my.sh"):
    #        #    open(fname+"/my.sh","w").close()
    #        #fname=fname+"/my.sh"
    #        #print fname
    #        if not nshfile%250:
    #            fout=open(fname+"/my_"+str(nshfile)+".sh","a+")
    #            #i+=1
    #            print nshfile
    #            fout.write(shcmdlist.pop())
    #            cmd = "bsub -q l "+"-o "+"rootfile.log "+'\"sh '+ fname+"/my.sh"+'\"'
    #            print cmd
    #            #os.system(cmd)
    #            #fout.truncate(0)
    #            #break

    #    elif len(shcmdlist)<250 and not len(shcmdlist)==0:
    #        #open(fname+"/my.sh","w").close()
    #        while(len(shcmdlist)):
    #            nshfile+=1
    #            i+=1
    #            #fname=fname+"/my"+".sh"
    #            fout=open(fname+"/my_"+str(nshfile)+".sh","a+")
    #            fout.write(shcmdlist.pop())
    #            if len(shcmdlist)==0:
    #                cmd = "bsub -q l "+"-o "+"rootfile.log "+'\"sh '+ fname+"/my.sh"+'\"'
    #                #os.system(cmd)
    #                print cmd
    #                fout.close()
    #print "nshfile ",nshfile









def ret_name(infile):
    infile = infile.split('.')
    infile = infile[0]+'.'+infile[1]
    return infile

def appendFilteredList(filteredlist,filepath):
    filename = filepath.split('/')[-1]
    suffix = filename.split('.')[-1]
    if (suffix==conf.FILESUFFIX):
        filteredlist.append(filepath)

def makeFilteredList(filteredlist,listdir):
    items = os.listdir(listdir)
    for items in items:
        newpath= listdir+'/'+item
        if os.path.isdir(newpath):
            makeFilteredList(filteredlist,newpath)
        if os.path.isfile(newpath):
            appendFilteredList(filteredlist,newpath)

