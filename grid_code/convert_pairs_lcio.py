#####################################
#
# convert pairs from CAIN to lcio format
# djeans 18 jan 2017
#
# initialize environment:
#  export PYTHONPATH=${LCIO}/src/python:${ROOTSYS}/lib
#
#####################################
import math
import random
from array import array

# --- LCIO dependencies ---
from pyLCIO import UTIL, EVENT, IMPL, IO, IOIMPL

#---- number of events per momentum bin -----

import sys, os, optparse, shutil, distutils
from distutils import dir_util
##############################
#  Parse command line input  #
##############################

usage = 'usage: %prog incoherent_pair.dat'
parser = optparse.OptionParser(usage=usage)
(options, args) = parser.parse_args()
if len(args) != 1:
    parser.error('Please enter pair file name')

inFilename=args[0]
#print ('args[0]')

base, ext = os.path.splitext(inFilename)
if not ext:
    print '%s has no file extension' % inFilename
    exit

print 'input file ',inFilename
print 'base name ',base



outDir = base
if os.path.exists(outDir):
    shutil.rmtree(outDir)
os.makedirs(str(base))
print base+'/ directory created'


#1processor = 'myProcessors'
#1#shutil.copytree(processor, outDir)
#1processorDir = outDir+'/'+processor
#1if not os.path.exists(processorDir):
#1    os.makedirs(processorDir)
#1distutils.dir_util.copy_tree(processor, processorDir)
#1print 'processor copied in '+ processorDir
#1
#1xml = 'dbd_500GeV.nung_1.xml'
#1shutil.copy(xml, outDir)
#1print 'xml copied in '+ outDir
#1
#1marlinFile = 'init_ilcsoft.sh'
#1shutil.copy(marlinFile, outDir)

outFilename = base+'/'+ base + os.extsep + 'slcio'

wrt = IOIMPL.LCFactory.getInstance().createLCWriter( )
wrt.open( outFilename , EVENT.LCIO.WRITE_NEW )
print " opened outfile: " , outFilename


# write a RunHeader
run = IMPL.LCRunHeaderImpl()
run.setRunNumber( 0 )
run.parameters().setValue("Generator","CAIN")
wrt.writeRunHeader( run )

col = IMPL.LCCollectionVec( EVENT.LCIO.MCPARTICLE )
evt = IMPL.LCEventImpl()

eventN=0
genstat = 1

evt.setEventNumber( eventN )
evt.addCollection( col , "MCParticle" )

inFile = open(inFilename, 'r')
print 'opened file ', inFilename

#j=0
for line in inFile:
    if 'NAME' in line:
        continue

    fields=line.split()

    if len(fields)!=15:
        print 'strange line with nentries', len(fields)
        continue

    k = int(fields[0])
    if k==2:
        pdg=11
        charge=-1
    elif k==3:
        pdg=-11
        charge=1
    else:
        print 'unknown particle, K=', fields[0]
        continue

    mass = 0.000511 # GeV

    time = float(fields[4])/2.98e8/1e-9 # convert from m to ns

    vx = float(fields[5]) / 1e-3 # m -> mm
    vy = float(fields[6]) / 1e-3 # m -> mm
    vz = float(fields[7]) / 1e-3 # m -> mm
    vertex  = array('d',[ vx, vy, vz ] )

    energy = float(fields[8]) / 1e9 # eV -> GeV

    px = float(fields[9] ) / 1e9  # eV -> GeV
    py = float(fields[10]) / 1e9  # eV -> GeV
    pz = float(fields[11]) / 1e9  # eV -> GeV
    momentum  = array('f',[ px, py, pz ] )

#--------------- create MCParticle -------------------

    mcp = IMPL.MCParticleImpl()

    mcp.setGeneratorStatus( genstat )
    mcp.setMass( mass )
    mcp.setPDG( pdg )
    mcp.setMomentum( momentum )
    mcp.setCharge( charge )
    mcp.setTime( time )
    mcp.setVertex( vertex )


#-------------------------------------------------------



#-------------------------------------------------------


    col.addElement( mcp )

wrt.writeEvent( evt )
#inFilename = inFilename.split('')
print base +'/'+inFilename+".slcio is created"
wrt.close()
