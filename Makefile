###################################################################
#  Programmer(s):  Daniel R. Reynolds @ SMU
###################################################################
#  Copyright (c) 2019, Southern Methodist University.
#  All rights reserved.
#  For details, see the LICENSE file.
###################################################################
#  ManyVector+MRIStep demonstration application Makefile
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

# listing of all test routines
TESTS = compile_test.exe \
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

# instruct Make to look in 'src' for source code files
VPATH = src

# target to build all test executables
all : ${TESTS}

# build rules for specific tests
linear_advection_x.exe : src/linear_advection.cpp euler3D.o io.o
	${CXX} ${CXXFLAGS} -DADVECTION_X ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@
linear_advection_y.exe : src/linear_advection.cpp euler3D.o io.o
	${CXX} ${CXXFLAGS} -DADVECTION_Y ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@
linear_advection_z.exe : src/linear_advection.cpp euler3D.o io.o
	${CXX} ${CXXFLAGS} -DADVECTION_Z ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@
sod_x.exe : src/sod.cpp euler3D.o io.o
	${CXX} ${CXXFLAGS} -DADVECTION_X ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@
sod_y.exe : src/sod.cpp euler3D.o io.o
	${CXX} ${CXXFLAGS} -DADVECTION_Y ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@
sod_z.exe : src/sod.cpp euler3D.o io.o
	${CXX} ${CXXFLAGS} -DADVECTION_Z ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@
hurricane_xy.exe : src/hurricane.cpp euler3D.o io.o
	${CXX} ${CXXFLAGS} -DTEST_XY ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@
hurricane_yz.exe : src/hurricane.cpp euler3D.o io.o
	${CXX} ${CXXFLAGS} -DTEST_YZ ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@
hurricane_zx.exe : src/hurricane.cpp euler3D.o io.o
	${CXX} ${CXXFLAGS} -DTEST_ZX ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

# general build rules
%.exe : %.o euler3D.o io.o
	${CXX} ${CXXFLAGS} ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

.cpp.o : include/euler3D.hpp
	${CXX} -c ${CXXFLAGS} ${OMPFLAGS} ${INCS} $< -o $@

outclean :
	\rm -rf diags*.txt output*.txt xslice*.png yslice*.png zslice*.png __pycache__

clean : outclean
	\rm -rf *.o *.orig

realclean : clean
	\rm -rf *.exe *.dSYM *~

####### End of Makefile #######
