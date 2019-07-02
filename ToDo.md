# Remaining items for "completion" of initial demonstration code (not necessarily in order of priority):

1. Fluid tests:

   a. simple linear advection [1D] ("entropy convection")  [done]

   b. SOD shock tube [1D] -- see Greenough & Rider JCP 2004  [done]

   c. Hurricane-like flow evolution with critical rotation [2D] (has
      exact solution) -- see Pan et al Num. Math. Theor. Meth. Appl. 2017,
      section 3.2.1-1  [done]

   d. Rayleigh-Taylor instability [2D] -- see Liska & Wendroff SISC
      2003 [in progress]

   e. Interacting bubbles -- see Yang and Cai SISC 2013, section 6.1.2
      [yet to begin] 

   f. Implosion problem [2D] -- see Liska & Wendroff SISC 2003, section
      4.7  [yet to begin] 

   g. Explosion problem [2D] -- see Liska & Wendroff SISC 2003, section
      4.8  [yet to begin] 

   h. Double mach reflection problem [2D] -- see Andreyev final report,
      section 4.2.2  [yet to begin] 

2. HDF5 parallel I/O, with restart capabilities.  Store current
   solutions as well as current time and step size to HDF5 files.
   Write out all current simulation parameters to a restart input
   file.  On restarts, create ARKStep solver as usual, but set initial
   step size to match stored value.

3. Chemical species:

   a. Add support for "nc" chemical species to be advected along with
      fluid (at fluid time scale); these should be initially stored in
      `N_Vector_Serial` on each process, and grouped along with
      `MPIManyVector`.  Include in this serial `N_Vector` a temperature 
      "correction" field, that begins each slow step with zero-valued
      initial condition, and that donates all energy back to fluid at
      each fluid RHS call.

   b. Add reaction network based on DIRK solver and dense/band direct
      solver.  Test this on its own by creating/evolving/freeing an
      ARKStep solver at each fluid time step, in a simple
      operator-split fashion. 

   c. Couple chemistry to fluid via MRIStep module.

   d. Convert chemical reaction network and inner DIRK solver to reside
      on GPU.  This will initially retain control structures on CPU,
      but do 'number crunching' for RHS, Jacobian and linear solves on
      GPU.

4. Viscosity: add fluid viscosity, treated implicitly with ARK method
   at slow time scale.  Precondition/solve the corresponding linear
   systems with our new scalable linear solver interfaces (e.g.,
   *hypre*, PETSc, Trilinos).  This must wait until MRIStep has
   capabilities for solving multirate problems with implicitness at
   the slow time scale (or we can run pure IMEX tests without
   chemistry).
     
