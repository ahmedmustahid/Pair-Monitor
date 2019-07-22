rm -rf build
mkdir build
cd build
cmake -C /cvmfs/ilc.desy.de/sw/x86_64_gcc49_sl6/v02-00-02/ILCSoft.cmake ..
make install
#cd ~/test
#Marlin dbd_500GeV.nung_1.xml
#cd myProcessors/
#Marlin  db
