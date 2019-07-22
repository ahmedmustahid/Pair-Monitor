#!/usr/bin/python
import os,conf
import subprocess
from util import *
runinfolist = getRunInfoList()
cmdstart="\"("
cmdend=" )\""

#initdir = "/home/belle2/mustahid/CAINPlus/CAINPlus/run/datasets"
for runinfo in runinfolist:
	cmdlist = []
	dirname = runinfo.getDirName()
	for startevt in range(len(runinfo.startevts)-1):
		subdirname = "ev_" + str(runinfo.startevts[startevt]) + "-" + str(runinfo.startevts[startevt+1])
		files = os.listdir(dirname+"/"+subdirname)
		for afile in files:
        		suffix = '.' + afile.split('.')[-1] # suffix = .i
			if suffix == ".i":
        			basename = afile.split(suffix)[0]
        			index = basename.split('_')[-1]
				startNEvent = runinfo.startevts[startevt]
				endNEvent = runinfo.startevts[startevt+1]
				if endNEvent > conf.nTotalEvents:
					endNEvent = conf.nTotalEvents
				#print str(isub) + " " + str(endNEvent)
        	                cmd = "cain+ " + dirname + '/' + subdirname + '/' + afile + " " + str(startNEvent) + " " + str(endNEvent)
				#print cmd
				cmdlist.append(cmd)


	nsubmit = 0
	while (1):
		if (len(cmdlist)<=0):
			break
		else:
			nsubmit += 1
			cmdlog = open(dirname + "/command_" + str(nsubmit) + ".log",'w')

			cmd = ""
			if len(cmdlist) > 0:
				cmd = cmdlist.pop()

			cmd = "bsub -q l " + "-o " + dirname + "/submit_" + str(nsubmit) + ".log " + " -e " + dirname + "/submit_" + str(nsubmit) + ".err " + "-J " + basename + "_submit_" + str(nsubmit) + " " + cmdstart + cmd + cmdend
			cmdlog.write(cmd)
			#print cmd
	        	subprocess.call(cmd,shell=True)
			#print "1 set end. nsubmit = " + str(nsubmit)

