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
  CXXFLAGS = -O3 -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
else
  CXXFLAGS = -O0 -g -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX -DDEBUG -fsanitize=address
endif

# check for OpenMP usage
ifeq ($(USEOMP),0)
  OMPFLAGS =
endif

# shortcuts for include and library paths, etc.
INCS = -I./src $(SUNINCDIRS) ${KLUINCDIRS}

SUNDLIBS = -L$(SUNLIBDIR) \
           -lsundials_arkode \
           -lsundials_nvecmpimanyvector \
           -lsundials_nvecparallel \
           -lsundials_nvecserial \
           -lsundials_sunlinsolklu
KLULIBS = -L$(KLULIBDIR) -lklu -lcolamd -lamd -lbtf -lsuitesparseconfig
LIBS = ${SUNDLIBS} ${KLULIBS} -lm

LDFLAGS = -Wl,-rpath,${SUNLIBDIR},-rpath,${KLULIBDIR}

# common source/object files on which all executables depend
COMMONSRC = euler3D.cpp utilities.cpp io.cpp gopt.c
COMMONOBJ = euler3D.o utilities.o io.o gopt.o

# listing of all test routines that use 'default' number of fields
TESTS = compile_test_fluid.exe \
        communication_test_fluid.exe \
        linear_advection_x.exe \
        linear_advection_y.exe \
        linear_advection_z.exe \
        sod_x.exe \
        sod_y.exe \
        sod_z.exe \
        hurricane_xy.exe \
        hurricane_zx.exe \
        hurricane_yz.exe \
        rayleigh_taylor.exe \
        interacting_bubbles.exe \
        implosion.exe \
        explosion.exe \
        double_mach_reflection.exe

# listing of all test routines that use a custom 'NVAR' value
COLORTESTS = compile_test_tracers.exe \
             communication_test_tracers.exe \
             hurricane_zx_color.exe

# instruct Make to look in 'src' for source code files
VPATH = src

# target to build all test executables
all : ${TESTS} ${COLORTESTS} buildclean

# general build rules
gopt.o : gopt.c gopt.h
	${CXX} ${CXXFLAGS} -c $<

%.exe : %.o ${COMMONOBJ}
	${CXX} ${CXXFLAGS} ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

.cpp.o : include/euler3D.hpp
	${CXX} -c ${CXXFLAGS} ${OMPFLAGS} ${INCS} $< -o $@

buildclean : 
	\rm -rf *.o

outclean :
	\rm -rf diags*.txt output*.txt xslice*.png yslice*.png zslice*.png __pycache__

clean : outclean buildclean
	\rm -rf *.orig *~ 

realclean : clean
	\rm -rf *.exe *.dSYM


# build rules for specific tests
compile_test_fluid.exe : compile_test.cpp ${COMMONSRC}
	\rm -rf *.o
	${CXX} ${CXXFLAGS} -DNVAR=5 ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@
	\rm -rf *.o

compile_test_tracers.exe : compile_test.cpp ${COMMONSRC}
	\rm -rf *.o
	${CXX} ${CXXFLAGS} -DNVAR=7 ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@
	\rm -rf *.o

communication_test_fluid.exe : communication_test.cpp utilities.cpp io.cpp gopt.c compile_test.cpp
	\rm -rf *.o
	${CXX} ${CXXFLAGS} -DNVAR=5 ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@
	\rm -rf *.o

communication_test_tracers.exe : communication_test.cpp utilities.cpp io.cpp gopt.c compile_test.cpp
	\rm -rf *.o
	${CXX} ${CXXFLAGS} -DNVAR=9 ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@
	\rm -rf *.o

linear_advection_x.exe : linear_advection.cpp ${COMMONOBJ}
	${CXX} ${CXXFLAGS} -DADVECTION_X ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

linear_advection_y.exe : linear_advection.cpp ${COMMONOBJ}
	${CXX} ${CXXFLAGS} -DADVECTION_Y ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

linear_advection_z.exe : linear_advection.cpp ${COMMONOBJ}
	${CXX} ${CXXFLAGS} -DADVECTION_Z ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

sod_x.exe : sod.cpp ${COMMONOBJ}
	${CXX} ${CXXFLAGS} -DADVECTION_X ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

sod_y.exe : sod.cpp ${COMMONOBJ}
	${CXX} ${CXXFLAGS} -DADVECTION_Y ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

sod_z.exe : sod.cpp ${COMMONOBJ}
	${CXX} ${CXXFLAGS} -DADVECTION_Z ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

hurricane_xy.exe : hurricane.cpp ${COMMONOBJ}
	${CXX} ${CXXFLAGS} -DTEST_XY ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

hurricane_yz.exe : hurricane.cpp ${COMMONOBJ}
	${CXX} ${CXXFLAGS} -DTEST_YZ ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

hurricane_zx.exe : hurricane.cpp ${COMMONOBJ}
	${CXX} ${CXXFLAGS} -DTEST_ZX ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

hurricane_zx_color.exe : hurricane.cpp ${COMMONSRC}
	\rm -rf *.o
	${CXX} ${CXXFLAGS} -DTEST_ZX -DNVAR=11 ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@
	\rm -rf *.o



####### End of Makefile #######
