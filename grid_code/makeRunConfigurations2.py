import os,re,shutil,conf,util
from util import *
from random import randrange



def runLoopOver(runinfo):
    global odd_val
    tempfilename='tmp.i'
    #Define input directory
    dirname = runinfo.getDirName()
    #print len(runinfo.startevts)-1
    for startevt in range(len(runinfo.startevts)-1):
        subdirname = 'ev_' + str(runinfo.startevts[startevt]) + '-' + str(runinfo.startevts[startevt+1])
        #print subdirname
        makeDirectory(dirname+'/'+subdirname)

        readfile = open(conf.inconf,'r')
        writefile = open(tempfilename,'w')
        odd_val = odd_val + 2
        for line in readfile:
            if line[0] == '!':
                writefile.write(line)
            else:
                #if re.search('FILE=',line):
                #    writefile.write(re.sub('FILE=','FILE=\''+dirname+'/'+subdirname+'/\'+',line))
                if re.search('Rand=12345',line):
                    #rand_odd_value = odd_val
                    #rand_odd_value = randrange(1,100000000,2)
                    writefile.write(re.sub('Rand=12345','Rand='+str(odd_val),line))
                else:
                    writefile.write(line)
        writefile.close()
        readfile.close()


        filename = dirname+'/'+subdirname+'/run' + runinfo.getRunID() +'_'+str(odd_val) +'.i'

        # copy the temporary file to "filename"
        shutil.copy(tempfilename,filename)

        #
        # step 2 ) Now we move to replace parameter values.
        #
        # loop over values
        for param in runinfo.params:

        	os.remove(tempfilename) # clear tempfilename
        	writefile = open(tempfilename,'w') # open new file
        	readfile = open(filename,'r') # open as read only

        	value = runinfo.values[runinfo.params.index(param)]

        	# Read input configuration
        	for line in readfile:

        		if line[0] == '!':
        		# Just copy if it is comment line
        			writefile.write(line)
        		else :

        			if re.search(param+'=',line):
        			# Replace values if we find the parameter name specified above.
        				parameterequal = param+'='
        				frag = line.split(parameterequal)[1]
        				# delimiter must be ',' or ';'
        				if frag.find(',') > 0:
        					frag = frag.split(',')[0]
        				elif frag.find(';') > 0:
        					frag = frag.split(';')[0]

        				# special treatment for meta characters
        				if frag.count('*') > 0:
        					frag = re.sub("\*","\\*",frag)
        				if frag.count('$') > 0:
        					frag = re.sub("\$","\\$",frag)
        				if frag.count('^') > 0:
        					frag = re.sub("\^","\\^",frag)
        				if frag.count('.') > 0:
        					frag = re.sub("\.","\\.",frag)
        				if frag.count('+') > 0:
        					frag = re.sub("\+","\\+",frag)

        				writefile.write(re.sub(frag,value,line))
        			else :
        			# Just copy unless we find the parameter name specified above
        				writefile.write(line)

        	readfile.close()
        	writefile.close()

        	shutil.copy(tempfilename,filename)

        # Clean up. Delete a temporary file
        os.remove(tempfilename)

        print filename + " is created."
        #return odd_val
        # Function end.

#def rand_odd():

def getFileNameList(runinfo):
    tempfilename='tmp.i'
    #Define input directory
    dirname = runinfo.getDirName()

    FileNameList = []
    for startevt in range(len(runinfo.startevts)-1):
        subdirname = 'ev_' + str(runinfo.startevts[startevt]) + '-' + str(runinfo.startevts[startevt+1])
        filenamepath = dirname+'/'+subdirname+'/'
        filenames = os.listdir(filenamepath)
        FileNameList.append(filename)
    return FileNameList

# Main function starts from here.
#

# Call functions

# getRunInfoList function will collect information
# from conf.py, store the data in an object (RunParameterSet),
# which is defined in conf.py, and eventually return the list of the objects.
runinfolist = getRunInfoList()
# loop over the list of RunParameterSet.
#for runinfo in runinfolist:
#	# Automatic directory name generation from parameters.
#	dirname = runinfo.getDirName()
#	# make run directories
#	makeDirectory(dirname)
#    odd_val = 9
#	runLoopOver(runinfo,odd_val)
#fnamelist = getFileNameList()
#
#for fname in fnamelist:
#    print fname

#getFilePathFromList()

global odd_val
odd_val = 7221
for runinfo in runinfolist:
    #print runinfo
    dirname=runinfo.getDirName()
    #odd_val = 7221
    #print('my name')
    #print(dirname)
    makeDirectory(dirname)
    #runLoopOver(runinfo,odd_val)
    runLoopOver(runinfo)

