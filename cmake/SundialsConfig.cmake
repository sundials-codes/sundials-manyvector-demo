# ---------------------------------------------------------------
# Programmer: Cody J. Balos @ LLNL
# ---------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2002-2019, Lawrence Livermore National Security
# and Southern Methodist University.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
# ---------------------------------------------------------------

# check for SUNDIALS location
if(NOT SUNDIALS_DIR)
  set(SUNDIALS_DIR "" CACHE FILEPATH "SUNDIALS install directory")
  message(FATAL_ERROR "SUNDIALS_DIR is not set")
else()
  set(SUNDIALS_DIR "${SUNDIALS_DIR}" CACHE FILEPATH "SUNDIALS install directory")
endif()

# determine SUNDIALS components needed
set(sundials_components "arkode;cvode;nvecmpimanyvector;nvecparallel;nvecserial;sunmatrixsparse;sunlinsolklu")
# if(ENABLE_CUDA)
#   list(APPEND sundials_components "nveccuda" "nvecmpiplusx")
# endif()
# if(ENABLE_RAJA)
#   list(APPEND sundials_components "nvecccudaraja" "nvecmpiplusx")
# endif()
# if(ENABLE_OpenMP)
#   list(APPEND sundials_components "nvecopenmp")
# endif()
# if(ENABLE_OpenMP_DEVICE)
#   list(APPEND sundials_components "nvecopenmpdev")
# endif()

foreach(component ${sundials_components})
  # find the library for the component
  find_library(${component}_LIBS sundials_${component}
    PATHS ${SUNDIALS_DIR}/lib ${SUNDIALS_DIR}/lib64
    NO_DEFAULT_PATH)
  if(${component}_LIBS)
    message(STATUS "Looking for SUNDIALS ${component}... success")
  else()
    message(FATAL_ERROR "Looking for SUNDIALS ${component}... failed")
  endif()

  # create target for the component
  if(NOT TARGET SUNDIALS::${component})
    add_library(SUNDIALS::${component} UNKNOWN IMPORTED)
    set_property(TARGET SUNDIALS::${component} PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${SUNDIALS_DIR}/include)
    set_property(TARGET SUNDIALS::${component} PROPERTY IMPORTED_LOCATION ${${component}_LIBS})
  endif()
endforeach()
