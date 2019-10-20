#!/usr/bin/python
import os,re,glob,conf
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
		dirname = "./CainFiles_Sigx/Run" + self.getRunID()
		#dirname = "./Run" + self.getRunID()
        #dirname = dirname + str(sample())
		return dirname
	def getDirName_grid(self):
		dirname = "Run" + self.getRunID()
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

#def getFileNameList_confy():
#    #Define input directory
#    runinfolist = getRunInfoList_confy()
#
#    FileNameList = []
#    for runinfo in runinfolist:
#
#        dirname = runinfo.getDirName()
#
#        for startevt in range(len(runinfo.startevts)-1):
#            subdirname = 'ev_' + str(runinfo.startevts[startevt]) + '-' + str(runinfo.startevts[startevt+1])
#            filenamepath = dirname+'/'+subdirname+'/'
#            #print(filenamepath)
#            #filename= [os.path.abspath(x) for x in os.listdir(filenamepath)]
#            #filename = os.path.join(filenamepath,filename)
#            filenames = [os.path.abspath(x) for x in glob.iglob(filenamepath+'*.i')]
#            #print(filenames)
#            for filename in filenames:
#                FileNameList.append(filename)
#    #print(FileNameList)
#    return FileNameList #List  of all files in a directory; not list of list, because all files are added from the lowest level




def getFilePathFromList():
    filenamelist = getFileNameList()
    print(filenamelist)
    #print('filenamelist:________')
    #print(filenamelist)
    #for filenames in filenamelist:
    #    print('filenames:_______')
    #    print(filenames)
    #    for filename in filenames:
    #        print('filename:_____')
    #        print(filename)
    #    #return filename



def ret_name(infile):
    infile = infile.split('.')
    infile = infile[0]+'.'+infile[1]
    return infile

