
source /cvmfs/ilc.desy.de/sw/x86_64_gcc49_sl6/v02-00-02/init_ilcsoft.sh
export MARLIN_DLL="$PWD/myProcessors/lib/libmymarlin.so:$MARLIN_DLL"
Marlin ./dbd_500GeV.nung_1.xml
