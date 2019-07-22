# Install script for directory: /home/belle2/mustahid/CAINPlus/CAINPlus/run/myProcessors

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/belle2/mustahid/CAINPlus/CAINPlus/run/myProcessors")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "RelWithDebInfo")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmymarlin.so.0.1.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmymarlin.so.0.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmymarlin.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "/home/belle2/mustahid/CAINPlus/CAINPlus/run/myProcessors/lib:/cvmfs/ilc.desy.de/sw/x86_64_gcc49_sl6/v02-00-02/Marlin/v01-16/lib:/cvmfs/ilc.desy.de/sw/x86_64_gcc49_sl6/v02-00-02/lcio/v02-12-01/lib:/cvmfs/ilc.desy.de/sw/x86_64_gcc49_sl6/v02-00-02/gear/v01-08/lib:/cvmfs/ilc.desy.de/sw/x86_64_gcc49_sl6/v02-00-02/CLHEP/2.3.4.3/lib:/cvmfs/ilc.desy.de/sw/x86_64_gcc49_sl6/ilcutil/v01-05/lib:/cvmfs/ilc.desy.de/sw/x86_64_gcc49_sl6/v02-00-02/root/6.08.06/lib")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE FILES
    "/home/belle2/mustahid/CAINPlus/CAINPlus/run/myProcessors/build/lib/libmymarlin.so.0.1.0"
    "/home/belle2/mustahid/CAINPlus/CAINPlus/run/myProcessors/build/lib/libmymarlin.so.0.1"
    "/home/belle2/mustahid/CAINPlus/CAINPlus/run/myProcessors/build/lib/libmymarlin.so"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmymarlin.so.0.1.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmymarlin.so.0.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmymarlin.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/cvmfs/ilc.desy.de/sw/x86_64_gcc49_sl6/v02-00-02/Marlin/v01-16/lib:/cvmfs/ilc.desy.de/sw/x86_64_gcc49_sl6/v02-00-02/lcio/v02-12-01/lib:/cvmfs/ilc.desy.de/sw/x86_64_gcc49_sl6/v02-00-02/gear/v01-08/lib:/cvmfs/ilc.desy.de/sw/x86_64_gcc49_sl6/v02-00-02/CLHEP/2.3.4.3/lib:/cvmfs/ilc.desy.de/sw/x86_64_gcc49_sl6/ilcutil/v01-05/lib:/cvmfs/ilc.desy.de/sw/x86_64_gcc49_sl6/v02-00-02/root/6.08.06/lib:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
           NEW_RPATH "/home/belle2/mustahid/CAINPlus/CAINPlus/run/myProcessors/lib:/cvmfs/ilc.desy.de/sw/x86_64_gcc49_sl6/v02-00-02/Marlin/v01-16/lib:/cvmfs/ilc.desy.de/sw/x86_64_gcc49_sl6/v02-00-02/lcio/v02-12-01/lib:/cvmfs/ilc.desy.de/sw/x86_64_gcc49_sl6/v02-00-02/gear/v01-08/lib:/cvmfs/ilc.desy.de/sw/x86_64_gcc49_sl6/v02-00-02/CLHEP/2.3.4.3/lib:/cvmfs/ilc.desy.de/sw/x86_64_gcc49_sl6/ilcutil/v01-05/lib:/cvmfs/ilc.desy.de/sw/x86_64_gcc49_sl6/v02-00-02/root/6.08.06/lib")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/belle2/mustahid/CAINPlus/CAINPlus/run/myProcessors/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
