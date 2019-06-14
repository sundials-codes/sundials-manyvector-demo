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
  CXXFLAGS = -O2 -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
else
  CXXFLAGS = -O0 -g -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
endif

# check for OpenMP usage
ifeq ($(USEOMP),0)
  OMPFLAGS =
endif

# shortcuts for include and library paths, etc.
INCS = -I. $(SUNINCDIRS) ${KLUINCDIRS}

SUNDLIBS = -L$(SUNLIBDIR) -lsundials_arkode -lsundials_nvecmpimanyvector -lsundials_nvecparallel -lsundials_nvecserial -lsundials_sunlinsolklu
KLULIBS = -L$(KLULIBDIR) -lklu -lcolamd -lamd -lbtf -lsuitesparseconfig
LIBS = ${SUNDLIBS} ${KLULIBS} -lm

LDFLAGS = -Wl,-rpath,${SUNLIBDIR},-rpath,${KLULIBDIR}

# listing of all test routines
TESTS = compile_test.exe

# target to build all test executables
all : ${TESTS}

# general build rules for C++ programs
%.exe : compile_test.o euler3D.o init_from_file.o
	${CXX} ${CXXFLAGS} ${OMPFLAGS} ${INCS} $^ ${LIBS} ${LDFLAGS} -o $@

.cpp.o : euler3d.hpp
	${CXX} -c ${CXXFLAGS} ${OMPFLAGS} ${INCS} $< -o $@

clean :
	\rm -rf *.o *.exe euler3D*.txt diags_euler3D.txt *.orig

realclean : clean
	\rm -rf *.dSYM *~ *.pyc

####### End of Makefile #######
