# ------------------------------------------------------------------------------
# Programmer(s):  David J. Gardner @ LLNL
# ------------------------------------------------------------------------------
# Copyright (c) 2020, Southern Methodist University and
# Lawrence Livermore National Security.
# All rights reserved.
# For details, see the LICENSE file.
# ------------------------------------------------------------------------------
# CMakeLists.txt for ManyVector+MRIStep demonstration application
# ------------------------------------------------------------------------------
# Note: any program that requires tracers/chemical species **must** specify the
# total number of variables per spatial location (NVAR) as a preprocessor
# directive. This number must be no smaller than 5 (rho, mx, my, mz, et); e.g.,
# to add two tracers per spatial location add the preprocessor directive
# "-DNVAR=7". To ensure that this value is consistent for an entire executable,
# it must be supplied to the compilation of all object files.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# create object library of common sources
# ------------------------------------------------------------------------------

# Create an object library for various NVAR values. Note the default NVAR value
# is 5 if it is not set by a compile definition.
set(_nvar_values 5 7 9 11 15)

foreach(nvar ${_nvar_values})

  # set the object library name based on the NVAR value
  set(_shared_obj shared_obj_nvar_${nvar})

  # create the object library
  add_library(${_shared_obj} OBJECT
    gopt.cpp
    io.cpp
    utilities.cpp)

  # add the NVAR compile definition
  target_compile_definitions(${_shared_obj}
    PUBLIC NVAR=${nvar})

  # set includes
  target_include_directories(${_shared_obj}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  # set link libraries
  target_link_libraries(${_shared_obj}
    PUBLIC
    MPI::MPI_CXX
    SUNDIALS::nvecserial
    SUNDIALS::nvecmpimanyvector
    SUNDIALS::nvecmanyvector
    SUNDIALS::sunmatrixsparse
    SUNDIALS::arkode)

  # add HDF5
  if(ENABLE_HDF5)
    target_include_directories(${_shared_obj}
      PUBLIC
      ${HDF5_INCLUDE_DIRS})
    target_link_libraries(${_shared_obj}
      PUBLIC
      ${HDF5_LIBRARIES}
      ${HDF5_HL_LIBRARIES})
    target_compile_definitions(${_shared_obj}
      PUBLIC
      USEHDF5)
  endif()

  # add RAJA
  target_link_libraries(${_shared_obj}
    PUBLIC
    RAJA
    Threads::Threads) # not sure why but Raja needs to link to Threads
  if(RAJA_BACKEND MATCHES "CUDA")
    target_link_libraries(${_shared_obj}
      PUBLIC
      SUNDIALS::nvecraja) # should be nveccudaraja but it's exported incorrectly
    target_compile_definitions(${_shared_obj}
      PUBLIC
      USE_DEVICE
      PUBLIC
      RAJA_CUDA)
  elseif(RAJA_BACKEND MATCHES "HIP")
    target_link_libraries(${_shared_obj}
      PUBLIC
      SUNDIALS::nvecraja) # should be nvechipraja but it's exported incorrectly
    target_compile_definitions(${_shared_obj}
      PUBLIC
      USE_DEVICE
      PUBLIC
      RAJA_HIP)
  elseif(RAJA_BACKEND MATCHES "OPENMP")
    target_link_libraries(${_shared_obj}
      PUBLIC
      SUNDIALS::nvecserial) # raja + OpenMP not supported in sundials yet
    target_compile_definitions(${_shared_obj}
      PUBLIC
      RAJA_OPENMP)
  elseif(RAJA_BACKEND MATCHES "SERIAL")
    target_link_libraries(${_shared_obj}
      PUBLIC
      SUNDIALS::nvecserial) # raja + serial not supported in sundials yet
    target_compile_definitions(${_shared_obj}
      PUBLIC
      RAJA_SERIAL)
  else()
    message(FATAL_ERROR "Unsupported RAJA backend: ${RAJA_BACKEND}")
  endif()

endforeach()

# set the linear solvers depending on the RAJA backend selected
set(_chem_solver raja_primordial_network.cpp)
if(RAJA_BACKEND MATCHES "SERIAL|OPENMP")
  set(_linear_solver SUNDIALS::sunmatrixsparse SUNDIALS::sunlinsolklu)
else()
  set(_linear_solver SUNDIALS::sunmatrixmagmadense SUNDIALS::sunlinsolmagmadense)
endif()


# ------------------------------------------------------------------------------
# Executables
# ------------------------------------------------------------------------------

# macro to create executables
macro(sundemo_add_executable target)

  # macro keyword arguments/options
  set(options )
  set(oneValueArgs )
  set(multiValueArgs SOURCES LINK_LIBRARIES COMPILE_DEFINITIONS)

  # parse keyword arguments/options
  cmake_parse_arguments(sundemo_add_executable
    "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # set source properties
  if (RAJA_BACKEND MATCHES "CUDA")
    foreach(file ${sundemo_add_executable_SOURCES})
      set_source_files_properties(${file} PROPERTIES LANGUAGE CUDA)
    endforeach()
  endif()

  # create executable target
  add_executable(${target}
    ${sundemo_add_executable_SOURCES})

  # set executable compile options
  if (RAJA_BACKEND MATCHES "CUDA")
    target_compile_options(${target}
      PRIVATE "--expt-extended-lambda")
  endif()

  # set executable link libraries
  target_link_libraries(${target}
    PRIVATE ${sundemo_add_executable_LINK_LIBRARIES})

  # set executable compile definitions
  target_compile_definitions(${target}
    PRIVATE ${sundemo_add_executable_COMPILE_DEFINITIONS})

  # install rule
  install(TARGETS ${target})

endmacro()

# fluid compile test
sundemo_add_executable(compile_test_fluid.exe
  SOURCES
    compile_test.cpp
    euler3D_main.cpp
  LINK_LIBRARIES
    shared_obj_nvar_5
  COMPILE_DEFINITIONS
    NVAR=5)

# tracers compile test
sundemo_add_executable(compile_test_tracers.exe
  SOURCES
    compile_test.cpp
    euler3D_main.cpp
  LINK_LIBRARIES
    shared_obj_nvar_7
  COMPILE_DEFINITIONS
    NVAR=7)

# fluid communication test
sundemo_add_executable(communication_test_fluid.exe
  SOURCES
    communication_test_main.cpp
    compile_test.cpp
  LINK_LIBRARIES
    shared_obj_nvar_5
  COMPILE_DEFINITIONS
    NVAR=5)

add_test(communication_test_fluid
  communication_test_fluid.exe)

# tracers communication test
sundemo_add_executable(communication_test_tracers.exe
  SOURCES
    communication_test_main.cpp
    compile_test.cpp
  LINK_LIBRARIES
    shared_obj_nvar_9
  COMPILE_DEFINITIONS
    NVAR=9)

add_test(communication_test_tracers
  communication_test_tracers.exe)

# fluid I/O test
sundemo_add_executable(io_test_fluid.exe
  SOURCES
    io_test_main.cpp
    compile_test.cpp
  LINK_LIBRARIES
    shared_obj_nvar_5
  COMPILE_DEFINITIONS
    NVAR=5)

add_test(io_test_fluid
  io_test_fluid.exe)

# tracers I/O test
sundemo_add_executable(io_test_tracers.exe
  SOURCES
    io_test_main.cpp
    compile_test.cpp
  LINK_LIBRARIES
    shared_obj_nvar_9
  COMPILE_DEFINITIONS
    NVAR=9)

add_test(io_test_tracers
  io_test_tracers.exe)

# linear advection tests
sundemo_add_executable(linear_advection_x.exe
  SOURCES
    linear_advection.cpp
    euler3D_main.cpp
  LINK_LIBRARIES
    shared_obj_nvar_5
  COMPILE_DEFINITIONS
    ADVECTION_X)

add_test(linear_advection_x
  linear_advection_x.exe -f ${CMAKE_SOURCE_DIR}/tests/linear_advection_x/input_linear_advection.txt)

sundemo_add_executable(linear_advection_y.exe
  SOURCES
    linear_advection.cpp
    euler3D_main.cpp
  LINK_LIBRARIES
    shared_obj_nvar_5
  COMPILE_DEFINITIONS
    ADVECTION_Y)

add_test(linear_advection_y
  linear_advection_y.exe -f ${CMAKE_SOURCE_DIR}/tests/linear_advection_y/input_linear_advection.txt)

sundemo_add_executable(linear_advection_z.exe
  SOURCES
    linear_advection.cpp
    euler3D_main.cpp
  LINK_LIBRARIES
    shared_obj_nvar_5
  COMPILE_DEFINITIONS
    ADVECTION_Z)

add_test(linear_advection_z
  linear_advection_z.exe -f ${CMAKE_SOURCE_DIR}/tests/linear_advection_z/input_linear_advection.txt)

# Rayleigh Taylor test
sundemo_add_executable(rayleigh_taylor.exe
  SOURCES
    rayleigh_taylor.cpp
    euler3D_main.cpp
  LINK_LIBRARIES
    shared_obj_nvar_5)

# Need to reduce test size for CI
# add_test(rayleigh_taylor
#   rayleigh_taylor.exe -f ${CMAKE_SOURCE_DIR}/tests/rayleigh_taylor/input_rayleigh_taylor.txt)

# sod tests
sundemo_add_executable(sod_x.exe
  SOURCES
    sod.cpp
    euler3D_main.cpp
  LINK_LIBRARIES
    shared_obj_nvar_5
  COMPILE_DEFINITIONS
    ADVECTION_X)

add_test(sod_x
  sod_x.exe -f ${CMAKE_SOURCE_DIR}/tests/sod_x/input_sod.txt)

sundemo_add_executable(sod_y.exe
  SOURCES
    sod.cpp
    euler3D_main.cpp
  LINK_LIBRARIES
    shared_obj_nvar_5
  COMPILE_DEFINITIONS
    ADVECTION_Y)

add_test(sod_y
  sod_y.exe -f ${CMAKE_SOURCE_DIR}/tests/sod_y/input_sod.txt)

sundemo_add_executable(sod_z.exe
  SOURCES
    sod.cpp
    euler3D_main.cpp
  LINK_LIBRARIES
    shared_obj_nvar_5
  COMPILE_DEFINITIONS
    ADVECTION_Z)

add_test(sod_z
  sod_z.exe -f ${CMAKE_SOURCE_DIR}/tests/sod_z/input_sod.txt)

# hurricane tests
sundemo_add_executable(hurricane_xy.exe
  SOURCES
    hurricane.cpp
    euler3D_main.cpp
  LINK_LIBRARIES
    shared_obj_nvar_5
  COMPILE_DEFINITIONS
    TEST_XZ)

add_test(hurricane_xy
  hurricane_xy.exe -f ${CMAKE_SOURCE_DIR}/tests/hurricane_xy/input_hurricane.txt)

sundemo_add_executable(hurricane_yz.exe
  SOURCES
    hurricane.cpp
    euler3D_main.cpp
  LINK_LIBRARIES
    shared_obj_nvar_5
  COMPILE_DEFINITIONS
    TEST_YZ)

add_test(hurricane_yz
  hurricane_yz.exe -f ${CMAKE_SOURCE_DIR}/tests/hurricane_yz/input_hurricane.txt)

sundemo_add_executable(hurricane_zx.exe
  SOURCES
    hurricane.cpp
    euler3D_main.cpp
  LINK_LIBRARIES
    shared_obj_nvar_5
  COMPILE_DEFINITIONS
    TEST_ZX)

add_test(hurricane_zx
  hurricane_zx.exe -f ${CMAKE_SOURCE_DIR}/tests/hurricane_zx/input_hurricane.txt)

sundemo_add_executable(hurricane_zx_color.exe
  SOURCES
    hurricane.cpp
    euler3D_main.cpp
  LINK_LIBRARIES
    shared_obj_nvar_11
  COMPILE_DEFINITIONS
    TEST_ZX
    NVAR=11)

add_test(hurricane_zx_color
  hurricane_zx_color.exe -f ${CMAKE_SOURCE_DIR}/tests/hurricane_zx_color/input_hurricane.txt)

# primordial ode tests
sundemo_add_executable(primordial_ode.exe
  SOURCES
    primordial_ode_main.cpp
    ${_chem_solver}
  LINK_LIBRARIES
    shared_obj_nvar_5
    ${_linear_solver}
  COMPILE_DEFINITIONS
    CVKLU
    MAX_NCELLS=1000000
    NTHREADS=1)

# primordial static tests
sundemo_add_executable(primordial_static_imex.exe
  SOURCES
    primordial_static.cpp
    imex_chem_hydro_main.cpp
    ${_chem_solver}
  LINK_LIBRARIES
    shared_obj_nvar_15
    ${_linear_solver}
  COMPILE_DEFINITIONS
    CVKLU
    MAX_NCELLS=1000000
    NTHREADS=1
    NVAR=15)

sundemo_add_executable(primordial_static_mr.exe
  SOURCES
    primordial_static.cpp
    multirate_chem_hydro_main.cpp
    ${_chem_solver}
  LINK_LIBRARIES
    shared_obj_nvar_15
    ${_linear_solver}
  COMPILE_DEFINITIONS
    CVKLU
    MAX_NCELLS=1000000
    NTHREADS=1
    NVAR=15)

# primordial blast tests
sundemo_add_executable(primordial_blast_imex.exe
  SOURCES
    primordial_blast.cpp
    imex_chem_hydro_main.cpp
    ${_chem_solver}
  LINK_LIBRARIES
    shared_obj_nvar_15
    ${_linear_solver}
  COMPILE_DEFINITIONS
    CVKLU
    MAX_NCELLS=1000000
    NTHREADS=1
    NVAR=15)

sundemo_add_executable(primordial_blast_mr.exe
  SOURCES
    primordial_blast.cpp
    multirate_chem_hydro_main.cpp
    ${_chem_solver}
  LINK_LIBRARIES
    shared_obj_nvar_15
    ${_linear_solver}
  COMPILE_DEFINITIONS
    CVKLU
    MAX_NCELLS=1000000
    NTHREADS=1
    NVAR=15)

# fluid blast test
sundemo_add_executable(fluid_blast.exe
  SOURCES
    fluid_blast.cpp
    euler3D_main.cpp
  LINK_LIBRARIES
    shared_obj_nvar_5
  COMPILE_DEFINITIONS
    NVAR=5)

add_test(fluid_blast
  fluid_blast.exe -f ${CMAKE_SOURCE_DIR}/tests/fluid_blast/input_fluid_blast.txt)

# magma scaling test
if(RAJA_BACKEND MATCHES "CUDA|HIP")
sundemo_add_executable(magma_scaling.exe
  SOURCES
    magma_scaling_main.cpp
  LINK_LIBRARIES
    shared_obj_nvar_15
    SUNDIALS::sunmatrixmagmadense
    SUNDIALS::sunlinsolmagmadense
  COMPILE_DEFINITIONS
    NTHREADS=1
    NVAR=15)
endif()
