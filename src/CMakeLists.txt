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
    SUNDIALS::nvecparallel
    SUNDIALS::nvecmpimanyvector
    SUNDIALS::sunmatrixsparse
    SUNDIALS::arkode
    SUNDIALS::cvode)

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
  if(ENABLE_RAJA)
    target_link_libraries(${_shared_obj}
      PUBLIC
      RAJA
      Threads::Threads) # not sure why but Raja needs to link to Threads
    target_compile_definitions(${_shared_obj}
      PUBLIC
      USERAJA)
    if(RAJA_BACKEND MATCHES "CUDA")
      target_link_libraries(${_shared_obj}
        PUBLIC
        SUNDIALS::nvecraja) # should be nveccudaraja but it's exported incorrectly
    elseif(RAJA_BACKEND MATCHES "HIP")
      target_link_libraries(${_shared_obj}
        PUBLIC
        SUNDIALS::nvecraja) # should be nvechipraja but it's exported incorrectly
    endif()
  endif()

endforeach()

# set the filename for chemistry routines depending on RAJA usage
if (ENABLE_RAJA)
  set(_chem_solver raja_primordial_network.cpp)
else()
  set(_chem_solver dengo_primordial_network.cpp)
endif()


# ------------------------------------------------------------------------------
# Executables
# ------------------------------------------------------------------------------

# fluid compile test
add_executable(compile_test_fluid.exe
  compile_test.cpp
  euler3D_main.cpp)
target_link_libraries(compile_test_fluid.exe PRIVATE shared_obj_nvar_5)
target_compile_definitions(compile_test_fluid.exe PRIVATE NVAR=5)
install(TARGETS compile_test_fluid.exe)

# tracers compile test
add_executable(compile_test_tracers.exe
  compile_test.cpp
  euler3D_main.cpp)
target_link_libraries(compile_test_tracers.exe PRIVATE shared_obj_nvar_7)
target_compile_definitions(compile_test_tracers.exe PRIVATE NVAR=7)
install(TARGETS compile_test_tracers.exe)

# fluid communication test
add_executable(communication_test_fluid.exe
  communication_test_main.cpp
  compile_test.cpp)
target_link_libraries(communication_test_fluid.exe PRIVATE shared_obj_nvar_5)
target_compile_definitions(communication_test_fluid.exe PRIVATE NVAR=5)
install(TARGETS communication_test_fluid.exe)

# tracers communication test
add_executable(communication_test_tracers.exe
  communication_test_main.cpp
  compile_test.cpp)
target_link_libraries(communication_test_tracers.exe PRIVATE shared_obj_nvar_9)
target_compile_definitions(communication_test_tracers.exe PRIVATE NVAR=9)
install(TARGETS communication_test_tracers.exe)

# fluid I/O test
add_executable(io_test_fluid.exe
  io_test_main.cpp
  compile_test.cpp)
target_link_libraries(io_test_fluid.exe PRIVATE shared_obj_nvar_5)
target_compile_definitions(io_test_fluid.exe PRIVATE NVAR=5)
install(TARGETS io_test_fluid.exe)

# tracers I/O test
add_executable(io_test_tracers.exe
  io_test_main.cpp
  compile_test.cpp)
target_link_libraries(io_test_tracers.exe PRIVATE shared_obj_nvar_9)
target_compile_definitions(io_test_tracers.exe PRIVATE NVAR=9)
install(TARGETS io_test_tracers.exe)

# linear advection tests
add_executable(linear_advection_x.exe
  linear_advection.cpp
  euler3D_main.cpp)
target_link_libraries(linear_advection_x.exe PRIVATE shared_obj_nvar_5)
target_compile_definitions(linear_advection_x.exe PRIVATE ADVECTION_X)
install(TARGETS linear_advection_x.exe)

add_executable(linear_advection_y.exe
  linear_advection.cpp
  euler3D_main.cpp)
target_link_libraries(linear_advection_y.exe PRIVATE shared_obj_nvar_5)
target_compile_definitions(linear_advection_y.exe PRIVATE ADVECTION_Y)
install(TARGETS linear_advection_y.exe)

add_executable(linear_advection_z.exe
  linear_advection.cpp
  euler3D_main.cpp)
target_link_libraries(linear_advection_z.exe PRIVATE shared_obj_nvar_5)
target_compile_definitions(linear_advection_z.exe PRIVATE ADVECTION_Z)
install(TARGETS linear_advection_z.exe)

# Rayleigh Taylor test
add_executable(rayleigh_taylor.exe
  rayleigh_taylor.cpp
  euler3D_main.cpp)
target_link_libraries(rayleigh_taylor.exe PRIVATE shared_obj_nvar_5)
install(TARGETS rayleigh_taylor.exe)

# sod tests
add_executable(sod_x.exe
  sod.cpp
  euler3D_main.cpp)
target_link_libraries(sod_x.exe PRIVATE shared_obj_nvar_5)
target_compile_definitions(sod_x.exe PRIVATE ADVECTION_X)
install(TARGETS sod_x.exe)

add_executable(sod_y.exe
  sod.cpp
  euler3D_main.cpp)
target_link_libraries(sod_y.exe PRIVATE shared_obj_nvar_5)
target_compile_definitions(sod_y.exe PRIVATE ADVECTION_Y)
install(TARGETS sod_y.exe)

add_executable(sod_z.exe
  sod.cpp
  euler3D_main.cpp)
target_link_libraries(sod_z.exe PRIVATE shared_obj_nvar_5)
target_compile_definitions(sod_z.exe PRIVATE ADVECTION_Z)
install(TARGETS sod_z.exe)

# hurricane tests
add_executable(hurricane_xy.exe
  hurricane.cpp
  euler3D_main.cpp)
target_link_libraries(hurricane_xy.exe PRIVATE shared_obj_nvar_5)
target_compile_definitions(hurricane_xy.exe PRIVATE TEST_XZ)
install(TARGETS hurricane_xy.exe)

add_executable(hurricane_yz.exe
  hurricane.cpp
  euler3D_main.cpp)
target_link_libraries(hurricane_yz.exe PRIVATE shared_obj_nvar_5)
target_compile_definitions(hurricane_yz.exe PRIVATE TEST_YZ)
install(TARGETS hurricane_yz.exe)

add_executable(hurricane_zx.exe
  hurricane.cpp
  euler3D_main.cpp)
target_link_libraries(hurricane_zx.exe PRIVATE shared_obj_nvar_5)
target_compile_definitions(hurricane_zx.exe PRIVATE TEST_ZX)
install(TARGETS hurricane_zx.exe)

add_executable(hurricane_zx_color.exe
  hurricane.cpp
  euler3D_main.cpp)
target_link_libraries(hurricane_zx_color.exe PRIVATE shared_obj_nvar_11)
target_compile_definitions(hurricane_zx_color.exe PRIVATE TEST_ZX NVAR=11)
install(TARGETS hurricane_zx_color.exe)

# primordial ode tests
add_executable(primordial_ode.exe
  primordial_ode_main.cpp
  ${_chem_solver})
target_link_libraries(primordial_ode.exe
  PRIVATE
  shared_obj_nvar_5
  SUNDIALS::sunlinsolklu)
target_compile_definitions(primordial_ode.exe
  PRIVATE CVKLU MAX_NCELLS=1000000 NTHREADS=1)
install(TARGETS primordial_ode.exe)

add_executable(primordial_ode_CVODE.exe
  primordial_ode_main.cpp
  ${_chem_solver})
target_link_libraries(primordial_ode_CVODE.exe
  PRIVATE
  shared_obj_nvar_5
  SUNDIALS::sunlinsolklu)
target_compile_definitions(primordial_ode_CVODE.exe
  PRIVATE CVKLU MAX_NCELLS=1000000 NTHREADS=1 USE_CVODE)
install(TARGETS primordial_ode_CVODE.exe)

# primordial static tests
add_executable(primordial_static_imex.exe
  primordial_static.cpp
  imex_chem_hydro_main.cpp
  ${_chem_solver})
target_link_libraries(primordial_static_imex.exe
  PRIVATE
  shared_obj_nvar_15
  SUNDIALS::sunlinsolklu)
target_compile_definitions(primordial_static_imex.exe
  PRIVATE CVKLU MAX_NCELLS=1000000 NTHREADS=1 NVAR=15)
install(TARGETS primordial_static_imex.exe)

add_executable(primordial_static_mr.exe
  primordial_static.cpp
  multirate_chem_hydro_main.cpp
  ${_chem_solver})
target_link_libraries(primordial_static_mr.exe
  PRIVATE
  shared_obj_nvar_15
  SUNDIALS::sunlinsolklu)
target_compile_definitions(primordial_static_mr.exe
  PRIVATE CVKLU MAX_NCELLS=1000000 NTHREADS=1 NVAR=15)
install(TARGETS primordial_static_mr.exe)

# primordial blast tests
add_executable(primordial_blast_imex.exe
  primordial_blast.cpp
  imex_chem_hydro_main.cpp
  ${_chem_solver})
target_link_libraries(primordial_blast_imex.exe
  PRIVATE
  shared_obj_nvar_15
  SUNDIALS::sunlinsolklu)
target_compile_definitions(primordial_blast_imex.exe
  PRIVATE CVKLU MAX_NCELLS=1000000 NTHREADS=1 NVAR=15)
install(TARGETS primordial_blast_imex.exe)

add_executable(primordial_blast_mr.exe
  primordial_blast.cpp
  multirate_chem_hydro_main.cpp
  ${_chem_solver})
target_link_libraries(primordial_blast_mr.exe
  PRIVATE
  shared_obj_nvar_15
  SUNDIALS::sunlinsolklu)
target_compile_definitions(primordial_blast_mr.exe
  PRIVATE CVKLU MAX_NCELLS=1000000 NTHREADS=1 NVAR=15)
install(TARGETS primordial_blast_mr.exe)

# fluid blast test
add_executable(fluid_blast.exe
  fluid_blast.cpp
  euler3D_main.cpp)
target_link_libraries(fluid_blast.exe PRIVATE shared_obj_nvar_5)
target_compile_definitions(fluid_blast.exe PRIVATE NVAR=5)
install(TARGETS fluid_blast.exe)
