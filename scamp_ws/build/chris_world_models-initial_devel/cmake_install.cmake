# Install script for directory: /home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/chris_world_models-initial_devel

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
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
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/build/chris_world_models-initial_devel/catkin_generated/installspace/chris_world_models.pc")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/chris_world_models/cmake" TYPE FILE FILES
    "/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/build/chris_world_models-initial_devel/catkin_generated/installspace/chris_world_modelsConfig.cmake"
    "/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/build/chris_world_models-initial_devel/catkin_generated/installspace/chris_world_modelsConfig-version.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/chris_world_models" TYPE FILE FILES "/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/chris_world_models-initial_devel/package.xml")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/chris_world_models" TYPE DIRECTORY FILES "/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/chris_world_models-initial_devel/worlds" REGEX "/[^/]*\\~$" EXCLUDE)
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/chris_world_models" TYPE DIRECTORY FILES "/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/chris_world_models-initial_devel/maps" REGEX "/[^/]*\\~$" EXCLUDE)
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/chris_world_models" TYPE DIRECTORY FILES "/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/chris_world_models-initial_devel/launch" REGEX "/[^/]*\\~$" EXCLUDE)
endif()

