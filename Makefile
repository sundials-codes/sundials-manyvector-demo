###################################################################
#  Programmer(s):  Daniel R. Reynolds @ SMU
###################################################################
#  Copyright (c) 2019, Southern Methodist University.
#  All rights reserved.
#  For details, see the LICENSE file.
###################################################################
#  ManyVector+MRIStep demonstration application Makefile
#
#  Note: any program that requires tracers/chemical species
#  **must** specify the total number of variables per spatial
#  location (NVAR) as a preprocessor directive.  This number
#  must be no smaller than 5 (rho, mx, my, mz, et); e.g., to
#  add two tracers per spatial location add the preprocessor
#  directive  "-DNVAR=7".  To ensure that this value is
#  consistent for an entire executable, it must be supplied to
#  the compilation of all object files -- this is handled
#  correctly in "compile_test.exe" below.
###################################################################

# read Machine-specific Makefile (if applicable)
include Makefile.in

# set C++ compilation flags based on build type
ifeq ($(LIBTYPE),OPT)
  OPTFLAGS = -O3
else
  OPTFLAGS = -O0 -g -DDEBUG -fsanitize=address
endif

# check for OpenMP usage
ifeq ($(USEOMP),0)
  OMPFLAGS =
endif

# shortcuts for include and library paths, etc.
SUNDLIBS = -L$(SUNLIBDIR) \
           -lsundials_arkode \
           -lsundials_cvode \
           -lsundials_nvecmpimanyvector \
           -lsundials_nvecparallel \
           -lsundials_nvecserial \
           -lsundials_sunmatrixsparse \
           -lsundials_sunlinsolklu
KLULIBS = -L$(KLULIBDIR) -lklu -lcolamd -lamd -lbtf -lsuitesparseconfig
HDFLIBS = -L$(HDFLIBDIR) -lhdf5 -lhdf5_hl -lsz
ifeq ($(USEHDF5),1)
  INCS = -I./src $(SUNINCDIRS) ${KLUINCDIRS} ${HDFINCDIRS}
  LIBS = ${SUNDLIBS} ${KLULIBS} ${HDFLIBS} -lm
  LDFLAGS = -Wl,-rpath,${SUNLIBDIR},-rpath,${KLULIBDIR},-rpath,${HDFLIBDIR}
  HDFFLAGS = -DUSEHDF5
else
  INCS = -I./src $(SUNINCDIRS) ${KLUINCDIRS}
  LIBS = ${SUNDLIBS} ${KLULIBS} -lm
  LDFLAGS = -Wl,-rpath,${SUNLIBDIR},-rpath,${KLULIBDIR}
  HDFFLAGS =
endif

# final version of compiler flags
CXXFLAGS = -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX ${OPTFLAGS} ${HDFFLAGS}

# common source/object files on which all executables depend
COMMONSRC = utilities.cpp io.cpp gopt.cpp
COMMONOBJ = utilities.o io.o gopt.o

# listing of all test routines
TESTS = communication_test_fluid.exe \
        communication_test_tracers.exe \
        compile_test_fluid.exe \
        compile_test_tracers.exe \
        hurricane_xy.exe \
        hurricane_yz.exe \
        hurricane_zx.exe \
        hurricane_zx_color.exe \
        io_test_fluid.exe \
        io_test_tracers.exe \
        linear_advection_x.exe \
        linear_advection_y.exe \
        linear_advection_z.exe \
        primordial_ode.exe \
        primordial_ode_CVODE.exe \
        primordial_static_imex.exe \
        primordial_static_mr.exe \
        rayleigh_taylor.exe \
        sod_x.exe \
        sod_y.exe \
        sod_z.exe \
        #interacting_bubbles.exe \
        #implosion.exe \
        #explosion.exe \
        #double_mach_reflection.exe

# instruct Make to look in 'src' for source code files
VPATH = src

# target to build all test executables
all : ${TESTS}

# general build rules
gopt.o : gopt.cpp gopt.hpp
	${CXX} ${CXXFLAGS} -c $<

.cpp.o : include/euler3D.hpp
	${CXX} -c ${CXXFLAGS} ${OMPFLAGS} ${INCS} $< -o $@

buildclean :
	\rm -rf *.o

outclean :
	\rm -rf diags*.txt restart_parameters.txt output*.hdf5 xslice*.png yslice*.png zslice*.png __pycache__

clean : outclean buildclean
	\rm -rf *.orig *~ */*~

realclean : clean
	\rm -rf *.exe *.dSYM


# build rules for specific tests
compile_test_fluid.exe : compile_test.cpp euler3D_main.cpp ${COMMONSRC}
	${CXX} ${CXXFLAGS} -DNVAR=5 ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

compile_test_tracers.exe : compile_test.cpp euler3D_main.cpp ${COMMONSRC}
	${CXX} ${CXXFLAGS} -DNVAR=7 ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

communication_test_fluid.exe : communication_test_main.cpp compile_test.cpp ${COMMONSRC}
	${CXX} ${CXXFLAGS} -DNVAR=5 ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

communication_test_tracers.exe : communication_test_main.cpp compile_test.cpp ${COMMONSRC}
	${CXX} ${CXXFLAGS} -DNVAR=9 ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

io_test_fluid.exe : io_test_main.cpp compile_test.cpp ${COMMONSRC}
	${CXX} ${CXXFLAGS} -DNVAR=5 ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

io_test_tracers.exe : io_test_main.cpp compile_test.cpp ${COMMONSRC}
	${CXX} ${CXXFLAGS} -DNVAR=9 ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

linear_advection_x.exe : linear_advection.cpp euler3D_main.o ${COMMONOBJ}
	${CXX} ${CXXFLAGS} -DADVECTION_X ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

linear_advection_y.exe : linear_advection.cpp euler3D_main.o ${COMMONOBJ}
	${CXX} ${CXXFLAGS} -DADVECTION_Y ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

linear_advection_z.exe : linear_advection.cpp euler3D_main.o ${COMMONOBJ}
	${CXX} ${CXXFLAGS} -DADVECTION_Z ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

rayleigh_taylor.exe : rayleigh_taylor.cpp euler3D_main.cpp ${COMMONSRC}
	${CXX} ${CXXFLAGS} ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

sod_x.exe : sod.cpp euler3D_main.o ${COMMONOBJ}
	${CXX} ${CXXFLAGS} -DADVECTION_X ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

sod_y.exe : sod.cpp euler3D_main.o ${COMMONOBJ}
	${CXX} ${CXXFLAGS} -DADVECTION_Y ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

sod_z.exe : sod.cpp euler3D_main.o ${COMMONOBJ}
	${CXX} ${CXXFLAGS} -DADVECTION_Z ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

hurricane_xy.exe : hurricane.cpp euler3D_main.o ${COMMONOBJ}
	${CXX} ${CXXFLAGS} -DTEST_XY ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

hurricane_yz.exe : hurricane.cpp euler3D_main.o ${COMMONOBJ}
	${CXX} ${CXXFLAGS} -DTEST_YZ ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

hurricane_zx.exe : hurricane.cpp euler3D_main.o ${COMMONOBJ}
	${CXX} ${CXXFLAGS} -DTEST_ZX ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

hurricane_zx_color.exe : hurricane.cpp euler3D_main.cpp ${COMMONSRC}
	${CXX} ${CXXFLAGS} -DTEST_ZX -DNVAR=11 ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

primordial_ode.exe : primordial_ode_main.cpp dengo_primordial_network.cpp ${COMMONOBJ}
	${CXX} ${CXXFLAGS} -DCVKLU -DMAX_NCELLS=1000000 -DNTHREADS=1 ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

primordial_ode_CVODE.exe : primordial_ode_main.cpp dengo_primordial_network.cpp ${COMMONOBJ}
	${CXX} ${CXXFLAGS} -DCVKLU -DMAX_NCELLS=1000000 -DNTHREADS=1 -DUSE_CVODE ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

primordial_static_imex.exe : primordial_static.cpp imex_chem_hydro_main.cpp dengo_primordial_network.cpp ${COMMONSRC}
	${CXX} ${CXXFLAGS} -DCVKLU -DMAX_NCELLS=1000000 -DNTHREADS=1 -DNVAR=15 ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

primordial_static_mr.exe : primordial_static.cpp multirate_chem_hydro_main.cpp dengo_primordial_network.cpp ${COMMONSRC}
	${CXX} ${CXXFLAGS} -DCVKLU -DMAX_NCELLS=1000000 -DNTHREADS=1 -DNVAR=15 ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@



####### End of Makefile #######
