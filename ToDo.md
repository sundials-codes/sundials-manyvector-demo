# Planned development tasks (not in order of priority):

#. Additional fluid tests:

   #. Rayleigh-Taylor instability [2D] -- see Liska & Wendroff SISC 2003

   #. Interacting bubbles -- see Yang and Cai SISC 2013, section 6.1.2

   #. Implosion problem [2D] -- see Liska & Wendroff SISC 2003, section 4.7

   #. Explosion problem [2D] -- see Liska & Wendroff SISC 2003, section 4.8

   #. Double mach reflection problem [2D] -- see Andreyev final report,
      section 4.2.2

#. Updated reactive flow tests:

   #. Transition current reaction data structures to "standard" GPU memory
      (i.e., not managed).

   #. Transition current reaction routines to also run on AMD GPUs (via HIP).

   #. Transition to a more "challenging" (and larger) system of reaction
      equations.

#. More refined profiling, either via a true profiling tool (e.g.,
   PAPI), or if we don't want to depend on many external modules, then
   manually-built via MPI utilities.

#. Add fluid viscosity, treated implicitly with ARK method
   at slow time scale.  Precondition/solve the corresponding linear
   systems with our new scalable linear solver interfaces (e.g.,
   *hypre*, PETSc, Trilinos).
