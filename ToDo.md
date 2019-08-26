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

2. Chemical species:

   a. Add support for "nchem" chemical species to be advected along with
      fluid (at fluid time scale); these should be initially stored in
      `N_Vector_Serial` on each process, and grouped along with
      `MPIManyVector`.  Include in this serial `N_Vector` a
      'duplicate' version of the energy/temperature, so that all
      chemistry calculations can eventually reside on the GPU. [done]

   b. Couple chemistry to fluid via existing MRIStep module. [in progress]

   c. 'Custom' inner time-stepper for MRIStep, so that fast time scale
      problems can be evolved separately (e.g., MPI process-local, or
      even more refined such as strips or individual cells).  [yet to begin]

   d. Convert chemical reaction network and inner DIRK solver to reside
      on GPU.  This will initially retain control structures on CPU,
      but do 'number crunching' for RHS, Jacobian and linear solves on
      GPU.  [yet to begin]

3. More refined profiling, either via a true profiling tool (e.g.,
   PAPI), or if we don't want to depend on many external modules, then
   manually-built via MPI utilities.  [yet to begin]

4. Viscosity: add fluid viscosity, treated implicitly with ARK method
   at slow time scale.  Precondition/solve the corresponding linear
   systems with our new scalable linear solver interfaces (e.g.,
   *hypre*, PETSc, Trilinos).  This must wait until MRIStep has
   capabilities for solving multirate problems with implicitness at
   the slow time scale (or we can run pure IMEX tests without
   chemistry).  [yet to begin]
