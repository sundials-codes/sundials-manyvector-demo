# ------------------------------------------------------------------------------
# Programmer(s):  David J. Gardner @ LLNL
# ------------------------------------------------------------------------------
# Copyright (c) 2019, Southern Methodist University and
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
# it must be supplied to the compilation of all object files -- this is handled
# correctly in "compile_test.exe" below.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# create object library of common sources
# ------------------------------------------------------------------------------

# >>>>>>> NEED TO UPDATE TO HAVE THE RIGHT NVAR VALUE <<<<<<< #
add_library(shared_obj OBJECT
  gopt.cpp
  io.cpp
  utilities.cpp)
target_include_directories(shared_obj
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(shared_obj
  PUBLIC
  MPI::MPI_CXX
  SUNDIALS::nvecserial
  SUNDIALS::nvecparallel
  SUNDIALS::nvecmpimanyvector
  SUNDIALS::sunmatrixsparse
  SUNDIALS::arkode
  SUNDIALS::cvode)

# add HDF5
if(ENABLE_HDF5)
  target_include_directories(shared_obj
    PUBLIC ${HDF5_INCLUDE_DIRS})
  target_link_libraries(shared_obj
    PUBLIC ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
  target_compile_definitions(shared_obj PUBLIC USEHDF5)
endif()

# ------------------------------------------------------------------------------
# Executables
# ------------------------------------------------------------------------------

# fluid compile test
add_executable(compile_test_fluid.exe
  compile_test.cpp
  euler3D_main.cpp)
target_link_libraries(compile_test_fluid.exe PRIVATE shared_obj)
target_compile_definitions(compile_test_fluid.exe PRIVATE NVAR=5)

# tracers compile test
add_executable(compile_test_tracers.exe
  compile_test.cpp
  euler3D_main.cpp)
target_link_libraries(compile_test_tracers.exe PRIVATE shared_obj)
target_compile_definitions(compile_test_tracers.exe PRIVATE NVAR=7)

# fluid communication test
add_executable(communication_test_fluid.exe
  communication_test_main.cpp
  compile_test.cpp)
target_link_libraries(communication_test_fluid.exe PRIVATE shared_obj)
target_compile_definitions(communication_test_fluid.exe PRIVATE NVAR=5)

# tracers communication test
add_executable(communication_test_tracers.exe
  communication_test_main.cpp
  compile_test.cpp)
target_link_libraries(communication_test_tracers.exe PRIVATE shared_obj)
target_compile_definitions(communication_test_tracers.exe PRIVATE NVAR=9)

# fluid I/O test
add_executable(io_test_fluid.exe
  io_test_main.cpp
  compile_test.cpp)
target_link_libraries(io_test_fluid.exe PRIVATE shared_obj)
target_compile_definitions(io_test_fluid.exe PRIVATE NVAR=5)

# tracers I/O test
add_executable(io_test_tracers.exe
  io_test_main.cpp
  compile_test.cpp)
target_link_libraries(io_test_tracers.exe PRIVATE shared_obj)
target_compile_definitions(io_test_tracers.exe PRIVATE NVAR=9)

# linear advection tests
add_executable(linear_advection_x.exe
  linear_advection.cpp
  euler3D_main.cpp)
target_link_libraries(linear_advection_x.exe PRIVATE shared_obj)
target_compile_definitions(linear_advection_x.exe PRIVATE ADVECTION_X)

add_executable(linear_advection_y.exe
  linear_advection.cpp
  euler3D_main.cpp)
target_link_libraries(linear_advection_y.exe PRIVATE shared_obj)
target_compile_definitions(linear_advection_y.exe PRIVATE ADVECTION_Y)

add_executable(linear_advection_z.exe
  linear_advection.cpp
  euler3D_main.cpp)
target_link_libraries(linear_advection_z.exe PRIVATE shared_obj)
target_compile_definitions(linear_advection_z.exe PRIVATE ADVECTION_Z)

# Rayleigh Taylor test
add_executable(rayleigh_taylor.exe
  rayleigh_taylor.cpp
  euler3D_main.cpp)
target_link_libraries(rayleigh_taylor.exe PRIVATE shared_obj)

# sod tests
add_executable(sod_x.exe
  sod.cpp
  euler3D_main.cpp)
target_link_libraries(sod_x.exe PRIVATE shared_obj)
target_compile_definitions(sod_x.exe PRIVATE ADVECTION_X)

add_executable(sod_y.exe
  sod.cpp
  euler3D_main.cpp)
target_link_libraries(sod_y.exe PRIVATE shared_obj)
target_compile_definitions(sod_y.exe PRIVATE ADVECTION_Y)

add_executable(sod_z.exe
  sod.cpp
  euler3D_main.cpp)
target_link_libraries(sod_z.exe PRIVATE shared_obj)
target_compile_definitions(sod_z.exe PRIVATE ADVECTION_Z)

# hurricane tests
add_executable(hurricane_xy.exe
  hurricane.cpp
  euler3D_main.cpp)
target_link_libraries(hurricane_xy.exe PRIVATE shared_obj)
target_compile_definitions(hurricane_xy.exe PRIVATE TEST_XZ)

add_executable(hurricane_yz.exe
  hurricane.cpp
  euler3D_main.cpp)
target_link_libraries(hurricane_yz.exe PRIVATE shared_obj)
target_compile_definitions(hurricane_yz.exe PRIVATE TEST_YZ)

add_executable(hurricane_zx.exe
  hurricane.cpp
  euler3D_main.cpp)
target_link_libraries(hurricane_zx.exe PRIVATE shared_obj)
target_compile_definitions(hurricane_zx.exe PRIVATE TEST_ZX)

add_executable(hurricane_zx_color.exe
  hurricane.cpp
  euler3D_main.cpp)
target_link_libraries(hurricane_zx_color.exe PRIVATE shared_obj)
target_compile_definitions(hurricane_zx_color.exe PRIVATE TEST_ZX)

# primordial ode tests
add_executable(primordial_ode.exe
  primordial_ode_main.cpp
  dengo_primordial_network.cpp)
target_link_libraries(primordial_ode.exe PRIVATE shared_obj)
target_compile_definitions(primordial_ode.exe
  PRIVATE CVKLU MAX_NCELLS=1000000 NTHREADS=1)

add_executable(primordial_ode_CVODE.exe
  primordial_ode_main.cpp
  dengo_primordial_network.cpp)
target_link_libraries(primordial_ode_CVODE.exe PRIVATE shared_obj)
target_compile_definitions(primordial_ode_CVODE.exe
  PRIVATE CVKLU MAX_NCELLS=1000000 NTHREADS=1 USE_CVODE)

# primordial static tests
add_executable(primordial_static_imex.exe
  primordial_static.cpp
  imex_chem_hydro_main.cpp
  dengo_primordial_network.cpp)
target_link_libraries(primordial_static_imex.exe PRIVATE shared_obj)
target_compile_definitions(primordial_static_imex.exe
  PRIVATE CVKLU MAX_NCELLS=1000000 NTHREADS=1 NVAR=15)
        
add_executable(primordial_static_mr.exe
  primordial_static.cpp
  multirate_chem_hydro_main.cpp
  dengo_primordial_network.cpp)
target_link_libraries(primordial_static_mr.exe PRIVATE shared_obj)
target_compile_definitions(primordial_static_mr.exe
  PRIVATE CVKLU MAX_NCELLS=1000000 NTHREADS=1 NVAR=15)