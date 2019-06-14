###################################################################
#  Programmer(s):  Daniel R. Reynolds @ SMU
###################################################################
#  Copyright (c) 2019, Southern Methodist University.
#  All rights reserved.
#  For details, see the LICENSE file.
###################################################################
#  ManyVector+MRIStep demonstration application Makefile
###################################################################

# flag to decide between optimized and debugging libraries
#LIBTYPE = DBG
LIBTYPE = OPT

# flag to enable/disable OpenMP (1=enable, 0=disable)
USEOMP = 0
OMPFLAGS =

# compilation flags, compiler/optimization string
ifeq ($(LIBTYPE),OPT)
  CFLAGS = -O2
  FFLAGS = -O2
  CXXFLAGS = -O2 -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
  COMPDIR = $(COMPILER)
else
  CFLAGS = -O0 -g
  FFLAGS = -O0 -g
  CXXFLAGS = -O0 -g -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
  COMPDIR = $(COMPILER)_debug
endif

# machine-specific compilers and installation locations
HOST=$(shell hostname)

#    longclaw
ifeq ($(HOST),longclaw)
  ifeq ($(LIBTYPE),OPT)
    INSTDIR = /home/dreynolds/research/Sundials/install_opt
  else
    INSTDIR = /home/dreynolds/research/Sundials/install_dbg
  endif
  KLUDIR = /usr/local/suitesparse-5.2.0
  KLULIBS = -L$(INSTDIR)/lib/ -lsundials_sunlinsolklu \
           -L$(KLUDIR)/gnu/lib -lklu -lcolamd -lamd -lbtf -lsuitesparseconfig \
           -lm
  KLUINCS = -I$(KLUDIR)/gnu/include/ -Wl,--as-needed
  CXX = /usr/local/mpich-3.2.1/gnu/bin/mpicxx --std=c++11
  ifeq ($(USEOMP),1)
    OMPFLAGS = -fopenmp
  endif
  LDFLAGS = -Wl,-rpath=${INSTDIR}/lib

#    cauchy
else ifeq ($(HOST),cauchy)
  ifeq ($(LIBTYPE),OPT)
    INSTDIR = /home/dreynolds/research/Sundials/install_opt
  else
    INSTDIR = /home/dreynolds/research/Sundials/install_dbg
  endif
  KLUDIR = /usr/local/suitesparse-5.2.0/gnu
  KLULIBS = -L$(INSTDIR)/lib/ -lsundials_sunlinsolklu \
           -L$(KLUDIR)/lib -lklu -lcolamd -lamd -lbtf -lsuitesparseconfig \
           -lm
  KLUINCS = -I$(KLUDIR)/include/ -Wl,--as-needed
  CXX = /usr/local/mpich-3.2.1/gnu/bin/mpicxx --std=c++11
  ifeq ($(USEOMP),1)
    OMPFLAGS = -fopenmp
  endif
  LDFLAGS = -Wl,-rpath=${INSTDIR}/lib

#    descartes
else ifeq ($(HOST),descartes.local)
  ifeq ($(LIBTYPE),OPT)
    INSTDIR = /Users/dreynolds/research/Sundials/install_opt
  else
    INSTDIR = /Users/dreynolds/research/Sundials/install_dbg
  endif
  KLUDIR = /usr/local/suite-sparse-5.3.0
  KLULIBS = -L$(INSTDIR)/lib/ -lsundials_sunlinsolklu \
           -L$(KLUDIR)/clang/lib -lklu -lcolamd -lamd -lbtf -lsuitesparseconfig \
           -lm
  KLUINCS = -I$(KLUDIR)/clang/include/
  CXX = /usr/local/mpich-3.3/clang/bin/mpicxx --std=c++11
  ifeq ($(USEOMP),1)
    OMPFLAGS = -Xpreprocessor -fopenmp -lomp
  endif
  LDFLAGS = -rpath ${INSTDIR}/lib

#    default
else
  ifeq ($(LIBTYPE),OPT)
    INSTDIR = /home/dreynolds/research/Sundials/install_opt
  else
    INSTDIR = /home/dreynolds/research/Sundials/install_dbg
  endif
  KLUDIR = /usr/local/suitesparse-4.5.3
  KLULIBS = -L$(INSTDIR)/lib/ -lsundials_sunlinsolklu \
           -L$(KLUDIR)/gnu/lib -lklu -lcolamd -lamd -lbtf -lsuitesparseconfig \
           -lm
  KLUINCS = -I$(KLUDIR)/gnu/include/ -Wl,--as-needed
  CXX = /usr/bin/mpicxx --std=c++11
  ifeq ($(USEOMP),1)
    OMPFLAGS = -fopenmp
  endif
  LDFLAGS = -Wl,-rpath=${INSTDIR}/lib

endif


# shortcuts for include and library paths, etc.
INCS = -I. -I$(INSTDIR)/include/ ${KLUINCS}
LIBS = -L$(INSTDIR)/lib \
       -lsundials_arkode \
       -lsundials_nvecmpimanyvector \
       -lsundials_nvecparallel \
       -lsundials_nvecserial \
       ${KLULIBS} -lm


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
	\rm -rf *.o *.exe diags_*.txt* *.orig

realclean : clean
	\rm -rf *.dSYM *~ *.pyc

####### End of Makefile #######
