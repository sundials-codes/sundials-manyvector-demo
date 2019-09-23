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

    parser.add_argument('outputfiles', type=str, nargs='+',
                        help='output files')

    parser.add_argument('--plotbreakdown', action='store_true',
                        help='plot breakdown of timings')

    parser.add_argument('--savefigs', action='store_true',
                        help='save figures to file')

    parser.add_argument('--debug', action='store_true',
                        help='enable debugging output')

    # parse command line args
    args = parser.parse_args()

    # create array of run data
    rundata = []

    for f in args.outputfiles:
        rundata.append(parseoutput(f))

    # sort run data by number of processes
    rundata.sort(key=sortnprocs)

    # print dictionary keys and values
    if (args.debug):
        for d in rundata:
            print "Output from:",d["file"]
            for key,val in d.items():
                if type(val) is dict:
                    for key2,val2 in val.items():
                        print key2, "=>", val2
                else:
                    print key, "=>", val
            print "----------"

    # get timing data
    if (args.plotbreakdown):
        for d in rundata:
            timings = d["timing"]

            keys = ["setup", "trans", "sim"]
            plotstackedbars(timings, keys, title="Total time breakdown", ylabel="seconds")

            keys = ["sim I/O", "sim MPI", "sim slow RHS", "sim fast RHS",
                    "sim fast Jac", "sim lsetup", "sim lsolve"]
            plotstackedbars(timings, keys, ylabel="seconds")

    # plot scaling data with setup time included
    keys = ["total", "sim", "trans", "setup"]
    plotscaling(rundata,                        # list with test dictionaries
                keys,                           # keys to plot
                filterkey=["fused ops", True],  # fiter key and value to include
                normalize=True,                 # normalie runtimes
                title="Fused Scaling",          # plot title
                save=args.savefigs,             # save figure to file or show
                fname="scaling_fused_with_setup.pdf")

    keys = ["total", "sim", "trans", "setup"]
    plotscaling(rundata,                        # list with test dictionaries
                keys,                           # keys to plot
                filterkey=["fused ops", False], # fiter key and value to include
                normalize=True,                 # normalie runtimes
                title="Unfused Scaling",        # plot title
                save=args.savefigs,             # save figure to file or show
                fname="scaling_unfused_with_setup.pdf")

    # plot scaling data with setup time removed
    keys = ["total", "sim", "trans"]
    plotscaling(rundata,                        # list with test dictionaries
                keys,                           # keys to plot
                filterkey=["fused ops", True],  # fiter key and value to include
                normalize=True,                 # normalie runtimes
                subtractkey=["setup", "total"], # subtract setup time from total
                title="Fused Scaling",          # plot title
                save=args.savefigs,             # save figure to file or show
                fname="scaling_fused_no_setup.pdf")

    keys = ["total", "sim", "trans"]
    plotscaling(rundata,                        # list with test dictionaries
                keys,                           # keys to plot
                filterkey=["fused ops", False], # fiter key and value to include
                normalize=True,                 # normalie runtimes
                subtractkey=["setup", "total"], # subtract setup time from total
                title="Unfused Scaling",        # plot title
                save=args.savefigs,             # save figure to file or show
                fname="scaling_unfused_no_setup.pdf")

    # compare scaling data with and without setup time
    keys = ["total", "sim", "trans", "setup" ]
    plotcomparescaling(rundata,                          # list with test dictionaries
                       keys,                             # keys to plot
                       filterkey=["fused ops", True],    # fiter key and value to include
                       normalize=True,                   # normalie runtimes
                       title="Fused vs Unfused Scaling", # plot title
                       save=args.savefigs,               # save figure to file or show
                       fname="scaling_compare_fused_and_unfused_with_setup.pdf")

    keys = ["total", "sim", "trans"]
    plotcomparescaling(rundata,                          # list with test dictionaries
                       keys,                             # keys to plot
                       filterkey=["fused ops", True],    # fiter key and value to include
                       normalize=True,                   # normalie runtimes
                       subtractkey=["setup", "total"],   # subtract setup time from total
                       title="Fused vs Unfused Scaling", # plot title
                       save=args.savefigs,               # save figure to file or show
                       fname="scaling_compare_fused_and_unfused_no_setup.pdf")

# ===============================================================================

def parseoutput(filename):

    # create empty dictionary for test
    test = {}

    # create empty dictionary for timings
    timing = {}

    # flags for parsing output file
    readtiming = False
    firstread  = True

    # add output file name to dictionary
    test["file"] = filename

    # parse output file
    with open(filename, 'r') as f:

        # read output file line by line
        for line in f:

            # split line into list
            text = shlex.split(line)

            if "nprocs:" in line:
                ntotal = int(text[1])
                nx     = int(text[2][1:])
                ny     = int(text[4])
                nz     = int(text[6][:-1])
                test["nprocs"] = [ntotal, nx, ny, nz]
                continue

            if "spatial domain:" in line:
                xlow  = float(text[2][1:-1])
                xhigh = float(text[3][:-1])
                ylow  = float(text[5][1:-1])
                yhigh = float(text[6][:-1])
                zlow  = float(text[8][1:-1])
                zhigh = float(text[9][:-1])
                test["domain"] = [ [xlow, xhigh], [ylow, yhigh], [zlow, zhigh] ]

            if "time domain:" in line:
                t0 = float(text[3][1:-1])
                tf = float(text[4][:-2])
                test["sim time"] = [t0, tf]
                t0 = float(text[6][1:-1])
                tf = float(text[7][:-1])
                test["sim time cgs"] = [t0, tf]

            if "slow timestep size:" in line:
                dt = float(text[-1])
                test["slow dt"] = dt

            if "fixed timestep size:" in line:
                dt = float(text[-1])
                test["fast dt"] = dt

            if "initial transient evolution:" in line:
                tt = float(text[-1])
                test["trans time"] = tt

            if "solution output" in line:
                if text[-1] == "disabled":
                    test["output"] = False
                else:
                    test["output"] = True

            if "bdry cond" in line:
                xlow  = int(text[6][1:-1])
                xhigh = int(text[7][:-1])
                ylow  = int(text[9][1:-1])
                yhigh = int(text[10][:-1])
                zlow  = int(text[12][1:-1])
                zhigh = int(text[13][:-1])
                test["bdry"] = [ [xlow, xhigh], [ylow, yhigh], [zlow, zhigh] ]

            if "gamma:" in line:
                test["gamma"] = float(text[-1])

            if "num chemical species:" in line:
                test["nspecies"] = int(text[-1])

            if "spatial grid:" in line:
                nx = int(text[2])
                ny = int(text[4])
                nz = int(text[6])
                test["mesh"] = [ nx, ny, nz ]

            if "fused N_Vector" in line:
                if text[-1] == "enabled":
                    test["fused ops"] = True
                else:
                    test["fused ops"] = False

            if "local N_Vector" in line:
                if text[-1] == "enabled":
                    test["local ops"] = True
                else:
                    test["local ops"] = False

            # skip lines without timings
            if "profiling results:" in line:
                readtiming = True
                continue
            elif not readtiming:
                continue

            # get timings
            avgtime = float(text[-10])
            mintime = float(text[-4])
            maxtime = float(text[-2])

            # add timing to dictionary
            if "Total setup" in line:
                if firstread:
                    timing["setup"] = [avgtime, mintime, maxtime]
                continue

            if "Total I/O" in line:
                if firstread:
                    timing["trans I/O"] = [avgtime, mintime, maxtime]
                else:
                    timing["sim I/O"] = [avgtime, mintime, maxtime]
                continue

            if "Total MPI" in line:
                if firstread:
                    timing["trans MPI"] = [avgtime, mintime, maxtime]
                else:
                    timing["sim MPI"] = [avgtime, mintime, maxtime]
                continue

            if "Total pack" in line:
                if firstread:
                    timing["trans pack"] = [avgtime, mintime, maxtime]
                else:
                    timing["sim pack"] = [avgtime, mintime, maxtime]
                continue

            if "Total flux" in line:
                if firstread:
                    timing["trans flux"] = [avgtime, mintime, maxtime]
                else:
                    timing["sim flux"] = [avgtime, mintime, maxtime]
                continue

            if "Total Euler" in line:
                if firstread:
                    timing["trans euler"] = [avgtime, mintime, maxtime]
                else:
                    timing["sim euler"] = [avgtime, mintime, maxtime]
                continue

            if "Total slow RHS" in line:
                if firstread:
                    timing["trans slow RHS"] = [avgtime, mintime, maxtime]
                else:
                    timing["sim slow RHS"] = [avgtime, mintime, maxtime]
                continue

            if "Total fast RHS" in line:
                if firstread:
                    timing["trans fast RHS"] = [avgtime, mintime, maxtime]
                else:
                    timing["sim fast RHS"] = [avgtime, mintime, maxtime]
                continue

            if "Total fast Jac" in line:
                if firstread:
                    timing["trans fast Jac"] = [avgtime, mintime, maxtime]
                else:
                    timing["sim fast Jac"] = [avgtime, mintime, maxtime]
                continue

            if "Total lsetup" in line:
                if firstread:
                    timing["trans lsetup"] = [avgtime, mintime, maxtime]
                else:
                    timing["sim lsetup"] = [avgtime, mintime, maxtime]
                continue

            if "Total lsolve" in line:
                if firstread:
                    timing["trans lsolve"] = [avgtime, mintime, maxtime]
                else:
                    timing["sim lsolve"] = [avgtime, mintime, maxtime]
                continue

            if "Total trans" in line:
                timing["trans"] = [avgtime, mintime, maxtime]
                readtiming = False
                firstread  = False
                continue

            if "Total sim" in line:
                timing["sim"] = [avgtime, mintime, maxtime]
                continue

            if "Total Total" in line:
                timing["total"] = [avgtime, mintime, maxtime]
                readtiming = False
                break

    test["timing"] = timing

    return test

# ===============================================================================

def plotstackedbars(data, keys, save = False, title=None, ylabel=None):

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

def plotscaling(rundata, keys, filterkey = [None, None], normalize = False,
                subtractkey = [None, None], title = None, save = False,
                fname = None):
    """
    Plot timing data as function of the number of processes

    Parameters:
    rundata:     list of test dictionaries to plot data from
    keys:        list of keys in timing dictionary of each test to plot
    filterkey:   filter test dictionaries, [ key to check, value to pass check ]
    normalize:   if True, normalize times by min value of first key in keys input
    subtractkey: subtract times from some or all keys in keys input
                 [ key to subtract, keys to subtract from (optional, default is all keys) ]
    title:       string to use for plot title
    save:        if True, save plot to file else show plot on screen
    """

    # check inputs
    if type(rundata) is not list:
        raise Exception('rundata must be a list, [ test1, test2, ...].')

    if type(keys) is not list:
        raise Exception('keys must be a list, [ key1, key2, ...].')

    if type(filterkey) is not list:
        raise Exception('filterkey must be a list, [ key, value ].')

    if len(filterkey) < 2:
        raise Exception('Too few values in filterkey, expected [ key, value ].')

    if type(subtractkey) is not list:
        raise Exception('subtractkey must be a list, [ key ] or [ key1, key2, ...].')

    # get times to remove
    subtractfrom = None
    if subtractkey[0]:
        stime = []
        for d in rundata:
            # check if tests are included/excluded based on a key
            if filterkey[0]:
                if d[filterkey[0]] == filterkey[1]:
                    stime.append(d["timing"][subtractkey[0]][0])
            else:
                stime.append(d["timing"][subtractkey[0]][0])
        stime = np.array(stime)
        # check if only certain keys should be subtracted from
        if len(subtractkey) > 1:
            subtractfrom = subtractkey[1:]

    # normalize based on the first key
    firstkey = True

    # get timing data for each key
    for k in keys:

        # clear list for timing data
        nprocs = []
        time   = []

        # extract timing data for each test
        for d in rundata:

            # check if tests are included/excluded based on a key
            if filterkey[0]:
                if d[filterkey[0]] == filterkey[1]:
                    nprocs.append(d["nprocs"][0])
                    time.append(d["timing"][k][0])
            else:
                nprocs.append(d["nprocs"][0])
                time.append(d["timing"][k][0])

        # convert lits to numpy array
        nprocs = np.array(nprocs)
        time   = np.array(time)

        # subtract time ### only subtract from some keys
        if subtractkey[0]:
            if subtractfrom:
                if k in subtractfrom:
                    time = time - stime
            else:
                time = time - stime

        # normalize run times
        if normalize:
            if firstkey:
                mintime  = np.amin(time)
                firstkey = False
            time = time / mintime

        # plot times
        plt.semilogx(nprocs, time, label=k)

    # add title, labels, legend, etc.
    if (title):
        plt.title(title)
    plt.xlabel("number of MPI processes")
    if (normalize):
        plt.ylabel("normalized run time (ref time = "+str(mintime)+" s)")
    else:
        plt.ylabel("run time (s)")
    plt.legend()
    plt.grid(linestyle="--")

    # save or show figure
    if (save):
        if (fname):
            plt.savefig(fname)
        else:
            plt.savefig("scaling.pdf")
        plt.close()
    else:
        plt.show()

# ===============================================================================

def plotcomparescaling(rundata, keys, filterkey, normalize = False,
                       subtractkey = [None, None], title = None, save = False,
                       fname = None):
    """
    Compare timing data as function of the number of processes for two setups

    Parameters:
    rundata:     list of test dictionaries to plot data from
    keys:        list of keys in timing dictionary of each test to plot
    filterkey:   filter defining groups to compare, [ key to check, value for group 1 ]
    normalize:   if True, normalize times by min value of first key in keys input
    subtractkey: subtract times from some or all keys in keys input
                 [ key to subtract, keys to subtract from (optional, default is all keys) ]
    title:       string to use for plot title
    save:        if True, save plot to file else show plot on screen
    """

    # check inputs
    if type(rundata) is not list:
        raise Exception('rundata must be a list, [ test1, test2, ...].')

    if type(keys) is not list:
        raise Exception('keys must be a list, [ key1, key2, ...].')

    if type(filterkey) is not list:
        raise Exception('filterkey must be a list, [ key, value ].')

    if len(filterkey) < 2:
        raise Exception('Too few values in filterkey, expected [ key, value ].')

    if type(subtractkey) is not list:
        raise Exception('subtractkey must be a list, [ key ] or [ key1, key2, ...].')

    # line colors
    color = ['#e41a1c','#377eb8','#4daf4a','#984ea3',
             '#ff7f00','#a65628','#f781bf','#999999','#000000']

    # get times to remove
    subtractfrom = None
    if subtractkey[0]:
        stime1 = []
        stime2 = []
        for d in rundata:
            if d[filterkey[0]] == filterkey[1]:
                stime1.append(d["timing"][subtractkey[0]][0])
            else:
                stime2.append(d["timing"][subtractkey[0]][0])
        stime1 = np.array(stime1)
        stime2 = np.array(stime2)
        # check if only certain keys should be subtracted from
        if len(subtractkey) > 1:
            subtractfrom = subtractkey[1:]

    # normalize based on the first key
    firstkey = True

    # loop counter
    i = 0

    # get timing data for each key
    for k in keys:

        # clear list for timing data
        nprocs1 = []
        time1   = []

        nprocs2 = []
        time2   = []

        # extract timing data for each test
        for d in rundata:

            # check if tests are included/excluded based on a key
            if d[filterkey[0]] == filterkey[1]:
                nprocs1.append(d["nprocs"][0])
                time1.append(d["timing"][k][0])
            else:
                nprocs2.append(d["nprocs"][0])
                time2.append(d["timing"][k][0])

        # convert lits to numpy array
        nprocs1 = np.array(nprocs1)
        time1   = np.array(time1)

        nprocs2 = np.array(nprocs2)
        time2   = np.array(time2)

        # subtract time ### only subtract from some keys
        if subtractkey[0]:
            if subtractfrom:
                if k in subtractfrom:
                    time1 = time1 - stime1
                    time2 = time2 - stime2
            else:
                time1 = time1 - stime1
                time2 = time2 - stime2

        # normalize run times
        if normalize:
            if firstkey:
                mintime = min(np.amin(time1), np.amin(time2))
                firstkey = False
            time1 = time1 / mintime
            time2 = time2 / mintime

        # plot times
        plt.semilogx(nprocs1, time1, label=k, color=color[i], linestyle='-')
        plt.semilogx(nprocs2, time2, label=k, color=color[i], linestyle='--')

        # update counter
        i += 1

    # add title, labels, legend, etc.
    if (title):
        plt.title(title)
    plt.xlabel("number of MPI processes")
    if (normalize):
        plt.ylabel("normalized run time (ref time = "+str(mintime)+" s)")
    else:
        plt.ylabel("run time (s)")
    plt.legend()
    plt.grid(linestyle="--")

    # save or show figure
    if (save):
        if (fname):
            plt.savefig(fname)
        else:
            plt.savefig("scaling.pdf")
        plt.close()
    else:
        plt.show()

# ===============================================================================

def sortnprocs(elem):
    return elem["nprocs"][0]

# ===============================================================================

if __name__ == "__main__":
    main()

# EOF
