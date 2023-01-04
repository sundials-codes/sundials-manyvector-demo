# ------------------------------------------------------------------------------
# Programmer(s): David J. Gardner @ LLNL
# ------------------------------------------------------------------------------
# Copyright (c) 2020, Southern Methodist University and
# Lawrence Livermore National Security.
# All rights reserved.
# For details, see the LICENSE file.
# ------------------------------------------------------------------------------

add_test(linear_advection_x
  ${CMAKE_BINARY_DIR}/src/linear_advection_x.exe -f ${CMAKE_SOURCE_DIR}/tests/linear_advection_x/input_linear_advection.txt)
