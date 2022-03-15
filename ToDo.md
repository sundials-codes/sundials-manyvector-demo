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

   b. Couple chemistry to fluid via existing MRIStep module. [done]

   c. 'Custom' inner time-stepper for MRIStep, so that fast time scale
      problems can be evolved separately (e.g., MPI process-local, or
      even more refined such as strips or individual cells).  [in progress]

      **Need to change how this is implemented:**

        - convert overall vector data structure to use N_VSerial component
          vectors for hydrodynamic variables (but retain MPIManyVector in
          general).  [done]

        - convert "fast" integrator to use a ManyVector with the same
          structure as the overall solution (including hydrodynamic
          variables), since at present the hydrodynamic variables are not
          being evolved at the fast time scale.  [yet to begin]

        - create a custom SUNLinearSolver for the rank-local problem that
          treats the Jacobian of the hydrodynamic variables as 0 (thus
          $I-gamma J = I$), and farms out the chemical solver to Magma.
          [yet to begin]

        - update node-local fast integrator to accept the MPIManyVector as
          input, grab the vector array, create ManyVector with those same
          subvectors, and then call ARKStep Evolve on that ManyVector.
          [yet to begin]

        - update "fast" rhs function to explicitly consider the MRI forcing
          data as an MPIManyVector array, but to consider its own input/output
          vectors as ManyVector.  Determine whether it needs to cast the
          components of the MPIManyVector as a ManyVector for this to work, or
          if it's call to the linear combination routine can work with a
          mixture of ManyVector and MPIManyVector arguments.  [yet to begin]

   d. Convert chemical reaction network and inner DIRK solver to reside
      on GPU.  This will initially retain control structures on CPU,
      but do 'number crunching' for RHS, Jacobian and linear solves on
      GPU.  [done]

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
