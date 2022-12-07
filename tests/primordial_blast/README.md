# Primordial Blast Test

This is the test problem that was used for the "multiphysics demonstration problem" from
[D.R. Reynolds, D.J. Gardner, C.S. Woodward and R. Chinomona, "ARKODE: A flexible IVP 
solver infrastructure for one-step methods," arXiv:2205.14077v1,
27 May 2022](https://doi.org/10.48550/arXiv.2205.14077).  The tests
for that section of the paper used the input files in the subdirectories "imex_scaling"
and "mr_scaling".


Input files:

* `input_primordial_blast_imex.txt` - run test with an IMEX integrator
* `input_primordial_blast_mr.txt` - run test with a multirate integrator
