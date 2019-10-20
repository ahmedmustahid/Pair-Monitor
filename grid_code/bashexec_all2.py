
import pprint
import os
from DIRAC.Core.Base import Script
Script.parseCommandLine()
from DIRAC import gLogger, S_OK, S_ERROR
from ILCDIRAC.Interfaces.API.NewInterface.UserJob import UserJob
from ILCDIRAC.Interfaces.API.NewInterface.Applications import GenericApplication
from ILCDIRAC.Interfaces.API.DiracILC import DiracILC

from DIRAC import gLogger, S_OK, S_ERROR
from ILCDIRAC.Interfaces.API.NewInterface.Applications import DDSim
import conf,util
from util import *
import subprocess

# ######################################
#class _Params():
#    def __init__(self):
#        self.isLocal = False
#        self.numberOfEvents = 100
#        self.inputFile = "/incoherent_pair/inco_pair_split.slcio"
#        self.outputFile = ""
#        self.outputDir = ""

class _Params():
    def __init__(self,nbevents, infile, outfile, outdir):
        self.isLocal = False
        self.numberOfEvents = nbevents
        self.inputFile = infile
        self.outputFile = outfile
        self.outputDir = outdir

    def setLocal( self, opt ):
        self.isLocal = True
        gLogger.info("Script is executed locally")
        return S_OK()

    def setNumberOfEvents( self, opt ):
        self.numberOfEvents = int(opt)
        gLogger.info("Number of events is %d" % self.numberOfEvents)
        return S_OK()

    def setInputFile( self, opt ):
        self.inputFile = opt
        gLogger.info("Input file is %s" % self.inputFile)
        return S_OK()

    def setOutputFile( self, opt ):
        self.outputFile = opt
        gLogger.info("Output file is %s" % self.outputFile)
        return S_OK()

    def setOutputDir( self, opt ):
        self.outputDir = opt
        gLogger.info("Output file is written at %s" % self.outputDir)
        return S_OK()

    def registerSwitches(self):
        Script.registerSwitch('l','local', 'If given, execute locally', self.setLocal )
        Script.registerSwitch('n:','number_of_events:', 'Number of events to simulate', self.setNumberOfEvents )
        Script.registerSwitch('i:', 'InputFile:', 'Input file name', self.setInputFile)
        Script.registerSwitch('f:', 'OutputFile:', 'Output file name', self.setOutputFile)
        Script.registerSwitch('w:', 'WriteDir:', 'Output directory. No output, if not given', self.setOutputDir)

        msg = '%s [options]\n' % Script.scriptName
        msg += 'Function: Submit ddsim job'
        Script.setUsageMessage(msg)

# ######################################
def subDDSim(clip1):

    # Decide parameters for a job
    outputSE = "KEK-SRM"

    isLocal = clip1.isLocal
    nbevts = 0 if clip1.numberOfEvents == 0 else clip1.numberOfEvents
    #print('inside subddsim(): nbevts ', nbevts)
    outputFile="" if clip1.outputFile == "" else clip1.outputFile
    #print('inside subddsim outfile ', outputFile)
    outputDir = clip1.outputDir
    #print('inside subddsim outdir ', outputDir)
    inputFile = clip1.inputFile
    #print('inside subddsim inputFile ', inputFile)
    if inputFile == "":
        gLogger.error("Input file for ddsim is not given.")
        exit(-1)

    # Create DIRAC objects for job submission

    dIlc = DiracILC()

    job = UserJob()
    job.setJobGroup( "myddsimjob" )
    job.setName( "myddsim" )
    job.setOutputSandbox(['*.log', '*.sh', '*.py', '*.xml'])
    job.setILDConfig("v02-00-02")

    # job.setInputSandbox(["a6-parameters.sin", "P2f_qqbar.sin"])
    # job.setDestination(["LCG.KEK.jp", "LCG.DESY-HH.de"])  # job submission destination
    job.setBannedSites(["LCG.UKI-SOUTHGRID-RALPP.uk"])      # a list of sites not to submit job
    # job.setCPUTime( cputime_limit_in_seconds_by_dirac_units )

    ddsim = DDSim()
    ddsim.setVersion("ILCSoft-02-00-02_gcc49")
    ddsim.setDetectorModel("ILD_l5_v05")
    ddsim.setInputFile(inputFile)
    ddsim.setNumberOfEvents(nbevts)
    extraCLIArguments =  " --steeringFile ddsim_steer_July26.py"
    extraCLIArguments += " --outputFile %s " % outputFile
    extraCLIArguments += " --vertexSigma 0.0 0.0 0.1968 0.0 --vertexOffset 0.0 0.0 0.0 0.0 "
    ddsim.setExtraCLIArguments( extraCLIArguments )

    return ddsim



def all_jobs(name):
    d= DiracILC(True,"repo.rep")

    ################################################
    j = UserJob()
    j.setJobGroup("PM1")
    j.setName("Exec1")
    banned_sites = ["OSG.BNL.us", "LCG.UKI-NORTHGRID-LIV-HEP.uk", "OSG.UCSDT2.us",
                            "LCG.SCOTGRIDDURHAM.uk", "LCG.NIKHEF.nl",
                                            "LCG.UKI-SOUTHGRID-RALPP.uk", "LCG.GRIF.fr", "LCG.Manchester.uk",
                                                            "LCG.UKI-LT2-IC-HEP.uk", "LCG.Weizmann.il"]


    j.setBannedSites(banned_sites)

    caindir = name
    #print('Cain directory is ',caindir)
    indata = ['LFN:/ilc/user/a/amustahid/cain.exe',str(caindir) ,'LFN:/ilc/user/a/amustahid/runcain.sh', 'LFN:/ilc/user/a/amustahid/convert_pairs_lcio.py', 'LFN:/ilc/user/a/amustahid/pyLCIO.tar.gz','/home/belle2/mustahid/useful/my.sh', './splitInput.py','./subddsim.py','./ddsim_steer_July26.py','./ILD_l5_v05.xml', './my2.sh','./dbd_500GeV.nung_1.xml','LFN:/ilc/user/a/amustahid/myProcessors.tar.gz','./create_dir.py', './conf.py','./util.py','./testcain.sh','./beam_250.i']
    j.setInputSandbox(indata)

    ################################################

    #app = GenericApplication()
    #app.setScript("create_dir.py")
    #app.setInputFile("testcain.sh")
    #logf = 'create_dir.log'
    #app.setLogFile(logf)
    #app.setDebug(debug=True)
    #create_dirname = 'create_dir'
    #app.setName(create_dirname)
    #res=j.append(app)
    #if not res['OK']:
    #  print res['Message']
    #  exit(1)
    ################################################
    appre = GenericApplication()
    name = name.split('/')
    #print(name)
    cain_name = name[-1]
    subdir = name[-2]
    dirname = name[-3]


    #print('Cain file name ', cain_name)
    appre.setScript("LFN:/ilc/user/a/amustahid/runcain.sh")
    #appre.setScript("testcain.sh")
    ifile = cain_name.split('.')
    ifile = ifile[0]+'.'+ifile[1]+'.'+ifile[2]
    #print('ifile ',ifile)


    appre.setArguments(ifile)
    #direc = 'LFN:/ilc/user/a/amustahid/'
    #appre.setInputFile(ifile+".i")
    #appre.setArguments("This is input arguments")
    logf = ifile+'_'+subdir+'.log'
    appre.setLogFile(logf)
    appre.setDebug(debug=True)
    name = 'CAIN'
    appre.setName(name)
    res=j.append(appre)
    if not res['OK']:
      print res['Message']
      exit(1)
    ################################################




    ################################################
    #appost = GenericApplication()
    #appost.setScript("myanal.sh")
    #appost.setArguments("This is my analysis step")
    #res=j.append(appost)
    #if not res['OK']:
    #  print res['Message']
    #  exit(1)


    ap = GenericApplication()
    ap.setScript('my.sh')
    logf = 'my.log'
    ap.setLogFile(logf)
    ap.setDebug(debug=True)
    name = 'my'
    ap.setName(name)
    res=j.append(ap)
    if not res['OK']:
      print res['Message']
      exit(1)


    outfile = 'incoherent_pair.dat'
    appre.setOutputFile(outfile)


    ################################################
    direc='incoherent_pair'
    inputFile=direc+'/'+'inco_pair_split.slcio'

   # global variables to hold command line parameters
    # ######################################
    base = '.'
    #outdir=base+'/'+dirname+'/slcio_test_2ndrun'
    outdir=base+'/'+dirname+'/Run_7'
    #print('outdir'+' '+str(outdir))
    geant_name = ifile
    outputFile= geant_name+'_'+subdir +'.slcio'

    #_clip = _Params(False,1,inputFile,outputFile,outdir)

    nbevents= 100
    clip = _Params(nbevents, inputFile, outputFile, outdir)
    ddsim = subDDSim(clip)
   ################################################

    res=j.append(ddsim)
    if not res['OK']:
      print res['Message']
      exit(1)

    j.setOutputData(outputFile,outdir,"KEK-SRM")
    j.setOutputSandbox(["*.log","*.dat","*.slcio"])
    j.dontPromptMe()
    res =  j.submit(d)
    #res = j.submit(d, mode='local')
    if res['OK']:
        print str(res["Value"])
        #print "Dirac job, "+str(res["Value"])+", was submitted."
    else:
        print "Failed to submit Dirac job. return message was as follows."
        pprint.pprint(res)




#_clip = _Params()
nbevents=0
inputFile =''
outputFile=''
outdir=''
_clip = _Params(nbevents, inputFile, outputFile, outdir)
_clip.registerSwitches()
Script.parseCommandLine()

filenamelist = getFileNameList()
cmd ="uname -a"
subprocess.call(cmd,shell=True)
#i=0
#print(filename)
#submitted_fname='/gpfs/home/belle2/mustahid/useful/CainFiles_Sigx/Run_SigxFactor0.8_SigyFactor1.0/ev_1-2/run_SigxFactor0.8_SigyFactor1.0_13.i'
#indx = filenamelist.index(submitted_fname)
#if i==indx:
#    print indx
#if i>indx:
#print len(filenamelist)
import random
random.shuffle(filenamelist)
for filename in filenamelist:
    #i = i +1
    all_jobs(filename)
    print(str(filename))
    #break
