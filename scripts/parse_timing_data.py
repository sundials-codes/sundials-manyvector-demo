#!/usr/bin/env python
# --------------------------------------------------------------------------
# Programmer(s): David J. Gardner @ LLNL
# --------------------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2002-2019, Lawrence Livermore National Security
# and Southern Methodist University.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
# --------------------------------------------------------------------------
# Script to parse and plot timing data
# --------------------------------------------------------------------------
# Useful arrays when plotting:
#
# color = ['#e41a1c','#377eb8','#4daf4a','#984ea3',
#          '#ff7f00','#a65628','#f781bf','#999999','#000000']
# colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c',
#           '#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6',
#           '#6a3d9a','#ffff99','#b15928','#000000']
# LineStyles = ['-','--','-.',':']
# MarkerStyles = ['.','o',
#                 'v','^','<','>',
#                 'd','D','s','p','h','H','8',
#                 '*',
#                 '+','x','1','2','3','4','|','_']
# --------------------------------------------------------------------------

import os, sys
import argparse
import shlex
import numpy as np
import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser(
        description='Plot error over time between two runs')

    parser.add_argument('outputfile', type=str,
                        help='output file name')

    # parse command line args
    args = parser.parse_args()

    # create  dictionary for timing data
    timings = parseoutput(args.outputfile)

    # print dictionary keys and values
    # for key,val in timings.items():
    #     print key, "=>", val

    keys = ["setup", "trans", "sim"]
    plotstackedbars(timings, keys, title="Total time breakdown", ylabel="seconds")

    keys = ["sim I/O", "sim MPI", "sim slow RHS", "sim fast RHS",
            "sim fast Jac", "sim lsetup", "sim lsolve"]
    plotstackedbars(timings, keys, ylabel="seconds")

# ===============================================================================

def parseoutput(filename):

    # creat empty dictionary for output data
    data = {}

    # flags for parsing output file
    readdata  = False
    firstread = True

    # parse output file
    with open(filename, 'r') as f:

        # read output file line by line
        for line in f:

            # skip lines without data
            if "profiling results:" in line:
                readdata = True
                continue
            elif not readdata:
                continue

            # split line into list
            text = shlex.split(line)

            # get data
            avgtime = float(text[-10])
            mintime = float(text[-4])
            maxtime = float(text[-2])

            # add data to dictionary
            if "Total setup" in line:
                if firstread:
                    data["setup"] = [avgtime, mintime, maxtime]
                continue

            if "Total I/O" in line:
                if firstread:
                    data["trans I/O"] = [avgtime, mintime, maxtime]
                else:
                    data["sim I/O"] = [avgtime, mintime, maxtime]
                continue

            if "Total MPI" in line:
                if firstread:
                    data["trans MPI"] = [avgtime, mintime, maxtime]
                else:
                    data["sim MPI"] = [avgtime, mintime, maxtime]
                continue

            if "Total pack" in line:
                if firstread:
                    data["trans pack"] = [avgtime, mintime, maxtime]
                else:
                    data["sim pack"] = [avgtime, mintime, maxtime]
                continue

            if "Total flux" in line:
                if firstread:
                    data["trans flux"] = [avgtime, mintime, maxtime]
                else:
                    data["sim flux"] = [avgtime, mintime, maxtime]
                continue

            if "Total Euler" in line:
                if firstread:
                    data["trans euler"] = [avgtime, mintime, maxtime]
                else:
                    data["sim euler"] = [avgtime, mintime, maxtime]
                continue

            if "Total slow RHS" in line:
                if firstread:
                    data["trans slow RHS"] = [avgtime, mintime, maxtime]
                else:
                    data["sim slow RHS"] = [avgtime, mintime, maxtime]
                continue

            if "Total fast RHS" in line:
                if firstread:
                    data["trans fast RHS"] = [avgtime, mintime, maxtime]
                else:
                    data["sim fast RHS"] = [avgtime, mintime, maxtime]
                continue

            if "Total fast Jac" in line:
                if firstread:
                    data["trans fast Jac"] = [avgtime, mintime, maxtime]
                else:
                    data["sim fast Jac"] = [avgtime, mintime, maxtime]
                continue

            if "Total lsetup" in line:
                if firstread:
                    data["trans lsetup"] = [avgtime, mintime, maxtime]
                else:
                    data["sim lsetup"] = [avgtime, mintime, maxtime]
                continue

            if "Total lsolve" in line:
                if firstread:
                    data["trans lsolve"] = [avgtime, mintime, maxtime]
                else:
                    data["sim lsolve"] = [avgtime, mintime, maxtime]
                continue

            if "Total trans" in line:
                data["trans"] = [avgtime, mintime, maxtime]
                readdata = False
                firstread = False
                continue

            if "Total sim" in line:
                data["sim"] = [avgtime, mintime, maxtime]
                continue

            if "Total Total" in line:
                data["total"] = [avgtime, mintime, maxtime]
                readdata = False
                break

    return data

# ===============================================================================

def plotstackedbars(data, keys, title=None, ylabel=None):

    # bar colors
    color = ['#e41a1c','#377eb8','#4daf4a','#984ea3',
             '#ff7f00','#a65628','#f781bf','#999999','#000000']

    # bar width and center
    width = 0.35
    center = 1 - width / 2.0

    # plot bars
    total = 0
    i = 0
    for k in keys:
        plt.bar(center, data[k][0], width, color=color[i], bottom=total)
        total += data[k][0]
        i += 1

    # add title, labels, legend, etc.
    if title:
        plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.legend(keys)
    plt.show()

# ===============================================================================

if __name__ == "__main__":
    main()

# EOF
