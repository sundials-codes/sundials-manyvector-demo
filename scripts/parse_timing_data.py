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

    parser.add_argument('--printtime', action='store_true',
                        help='print runtimes')

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
    if args.debug:
        for d in rundata:
            print "Output from:",d["file"]
            for key in sorted(d.keys()):
                if type(d[key]) is dict:
                    for key2 in sorted(d[key].keys()):
                        print key2, "=>", d[key][key2]
                else:
                    print key, "=>", d[key]
            print "----------"

    # get timing data
    if args.plotbreakdown:
        for d in rundata:
            timings = d["timing"]

            keys = ["setup", "trans", "sim"]
            plotstackedbars(timings, keys, title="Total time breakdown", ylabel="seconds")

            keys = ["sim I/O", "sim MPI", "sim slow RHS", "sim fast RHS",
                    "sim fast Jac", "sim lsetup", "sim lsolve"]
            plotstackedbars(timings, keys, ylabel="seconds")

    # plot scaling data with setup time removed
    keys = ["total w/o setup", "sim", "trans"]
    plotcomparescaling(rundata,                           # list with test dictionaries
                       keys,                              # keys to plot
                       filterkey=["fused ops", True],     # fiter key and value to include
                       normalize=True,                    # normalize runtimes
                       title="Scaling: Fused vs Unfused", # plot title
                       save=args.savefigs,                # save figure to file or show
                       fname="scaling_compare_fused_vs_unfused_no_setup.pdf")

    # total times without setup time
    keys = ["total w/o setup",
            "total fast RHS", "total sundials", "total lsolve"]
    plotcomparescaling(rundata,                           # list with test dictionaries
                       keys,                              # keys to plot
                       filterkey=["fused ops", True],     # fiter key and value to include
                       normalize=True,                    # normalize runtimes
                       errorbars=True,                    # add error bars to plot
                       title="Scaling: Fused vs Unfused", # plot title
                       labels=["total fused",
                               "total unfused",
                               "fast RHS fused",
                               "fast RHS unfused",
                               "Sundials fused",
                               "Sundials unfused",
                               "LSolve fused",
                               "LSolve unfused"],
                       save=args.savefigs,                # save figure to file or show
                       fname="scaling_total_fused_vs_unfused_no_setup.pdf")

    # total fused times without setup time
    keys = ["total w/o setup", "total fast RHS", "total sundials", "total lsolve", "total slow RHS"]
    plotscaling(rundata,                           # list with test dictionaries
                keys,                              # keys to plot
                filterkey=["fused ops", True],     # filter key and value to include
                normalize=True,                    # normalize runtimes
                errorbars=False,                   # add error bars to plot
                title="Weak Scaling: Multirate Compressible Reacting Flow",   # plot title
                Labels=["total",
                        "fast RHS",
                        "overhead",
                        "fast LSolve",
                        "slow RHS"],
                save=args.savefigs,                # save figure to file or show
                fname="scaling_fused_no_setup.pdf")

    # total unfused times without setup time
    keys = ["total w/o setup", "total fast RHS", "total sundials", "total lsolve", "total slow RHS"]
    plotscaling(rundata,                           # list with test dictionaries
                keys,                              # keys to plot
                filterkey=["fused ops", False],    # filter key and value to include
                normalize=True,                    # normalize runtimes
                errorbars=False,                   # add error bars to plot
                title="Weak Scaling: Multirate Compressible Reacting Flow",   # plot title
                Labels=["total",
                        "fast RHS",
                        "overhead",
                        "fast LSolve",
                        "slow RHS"],
                save=args.savefigs,                # save figure to file or show
                fname="scaling_unfused_no_setup.pdf")

    if args.printtime:
        keys = ["total w/o setup",
                "total I/O", "total MPI", "total pack", "total flux", "total euler",
                "total slow RHS", "total fast RHS", "total fast Jac", "total lsetup",
                "total lsolve", "total sundials"]
        printpercenttime(rundata,                          # list with test dictionaries
                         "total w/o setup",                # key for total time
                         keys,                             # keys to plot
                         filterkey=["fused ops", True],    # fiter key and value to include
                         normalize=True,                   # normalie runtimes
                         errorbars=True)                   # add error bars to plot

# ===============================================================================

def parseoutput(filename, minmax = "minvar", wminmax = False):

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

    # compute total time without setup
    time = []
    time.append(timing["total"][0] - timing["setup"][0]) # avg = avg - avg
    if minmax == "skewmax":
        # largest min time, largest max time: more variability in max time
        time.append(timing["total"][1] - timing["setup"][1]) # min = min - min
        time.append(timing["total"][2] - timing["setup"][1]) # max = max - min
    elif minmax == "skewmin":
        # smallest min time, smallest max time: more variability in min time
        time.append(timing["total"][1] - timing["setup"][2]) # min = min - max
        time.append(timing["total"][2] - timing["setup"][2]) # max = max - max
    elif minmax == "minvar":
        # largest min time, smallest max time: least variability
        time.append(timing["total"][1] - timing["setup"][1]) # min = min - min
        time.append(timing["total"][2] - timing["setup"][2]) # max = max - max
    elif minmax == "maxvar":
        # smallest min time, largest max time: most variability
        time.append(timing["total"][1] - timing["setup"][2]) # min = min - max
        time.append(timing["total"][2] - timing["setup"][1]) # max = max - min
    else:
        raise Exception('unknown minmax option: skewmax, skewmin, minvar, maxvar')

    # check if min is larger than avg
    if time[1] > time[0]:
        if wminmax:
            print "Warning: total time w/o setup min > avg for file:",filename
            print "min = ",time[1]
            print "avg = ",time[0]
            print "Setting min = avg"
        time[1] = time[0]
    # check if max is smaller than avg
    if time[2] < time[0]:
        if wminmax:
            print "Warning: total time w/o setup max < avg for file:",filename
            print "max = ",time[2]
            print "avg = ",time[0]
            print "Setting max = avg"
        time[2] = time[0]

    timing["total w/o setup"] = time

    # compute SUNDIALS trans time
    time = []
    time.append(timing["trans"][0] -
                (timing["trans slow RHS"][0] +
                 timing["trans fast RHS"][0] +
                 timing["trans fast Jac"][0] +
                 timing["trans lsetup"][0] +
                 timing["trans lsolve"][0]))

    if minmax == "skewmax":
        # largest min time, largest max time: more variability in max time
        time.append(timing["trans"][1] -           # min = min - min
                    (timing["trans slow RHS"][1] +
                     timing["trans fast RHS"][1] +
                     timing["trans fast Jac"][1] +
                     timing["trans lsetup"][1] +
                     timing["trans lsolve"][1]))
        time.append(timing["trans"][2] -           # max = max - min
                    (timing["trans slow RHS"][1] +
                     timing["trans fast RHS"][1] +
                     timing["trans fast Jac"][1] +
                     timing["trans lsetup"][1] +
                     timing["trans lsolve"][1]))
    elif minmax == "skewmin":
        # smallest min time, smallest max time: more variability in min time
        time.append(timing["trans"][1] -           # min = min - max
                    (timing["trans slow RHS"][2] +
                     timing["trans fast RHS"][2] +
                     timing["trans fast Jac"][2] +
                     timing["trans lsetup"][2] +
                     timing["trans lsolve"][2]))
        time.append(timing["trans"][2] -           # max = max - max
                    (timing["trans slow RHS"][2] +
                     timing["trans fast RHS"][2] +
                     timing["trans fast Jac"][2] +
                     timing["trans lsetup"][2] +
                     timing["trans lsolve"][2]))

    elif minmax == "minvar":
        # largest min time, smallest max time: least variability
        time.append(timing["trans"][1] -           # min = min - min
                    (timing["trans slow RHS"][1] +
                     timing["trans fast RHS"][1] +
                     timing["trans fast Jac"][1] +
                     timing["trans lsetup"][1] +
                     timing["trans lsolve"][1]))
        time.append(timing["trans"][2] -           # max = max - max
                    (timing["trans slow RHS"][2] +
                     timing["trans fast RHS"][2] +
                     timing["trans fast Jac"][2] +
                     timing["trans lsetup"][2] +
                     timing["trans lsolve"][2]))

    elif minmax == "maxvar":
        # smallest min time, largest max time: most variability
        time.append(timing["trans"][1] -           # min = min - max
                    (timing["trans slow RHS"][2] +
                     timing["trans fast RHS"][2] +
                     timing["trans fast Jac"][2] +
                     timing["trans lsetup"][2] +
                     timing["trans lsolve"][2]))
        time.append(timing["trans"][2] -           # max = max - min
                    (timing["trans slow RHS"][1] +
                     timing["trans fast RHS"][1] +
                     timing["trans fast Jac"][1] +
                     timing["trans lsetup"][1] +
                     timing["trans lsolve"][1]))
    else:
        raise Exception('unknown minmax option: skewmax, skewmin, minvar, maxvar')

    # check if min is larger than avg
    if time[1] > time[0]:
        if wminmax:
            print "Warning: trans sundials min > avg for file:",filename
            print "min = ",time[1]
            print "avg = ",time[0]
            print "Setting min = avg"
        time[1] = time[0]
    # check if max is smaller than avg
    if time[2] < time[0]:
        if wminmax:
            print "Warning: trans sundials max < avg for file:",filename
            print "max = ",time[2]
            print "avg = ",time[0]
            print "Setting max = avg"
        time[2] = time[0]

    timing["trans sundials"] = time

    # compute SUNDIALS sim time
    time = []
    time.append(timing["sim"][0] -
                (timing["sim slow RHS"][0] +
                 timing["sim fast RHS"][0] +
                 timing["sim fast Jac"][0] +
                 timing["sim lsetup"][0] +
                 timing["sim lsolve"][0]))

    if minmax == "skewmax":
        # largest min time, largest max time: more variability in max time
        time.append(timing["sim"][1] -           # min = min - min
                    (timing["sim slow RHS"][1] +
                     timing["sim fast RHS"][1] +
                     timing["sim fast Jac"][1] +
                     timing["sim lsetup"][1] +
                     timing["sim lsolve"][1]))
        time.append(timing["sim"][2] -           # max = max - min
                    (timing["sim slow RHS"][1] +
                     timing["sim fast RHS"][1] +
                     timing["sim fast Jac"][1] +
                     timing["sim lsetup"][1] +
                     timing["sim lsolve"][1]))
    elif minmax == "skewmin":
        # smallest min time, smallest max time: more variability in min time
        time.append(timing["sim"][1] -           # min = min - max
                    (timing["sim slow RHS"][2] +
                     timing["sim fast RHS"][2] +
                     timing["sim fast Jac"][2] +
                     timing["sim lsetup"][2] +
                     timing["sim lsolve"][2]))
        time.append(timing["sim"][2] -           # max = max - max
                    (timing["sim slow RHS"][2] +
                     timing["sim fast RHS"][2] +
                     timing["sim fast Jac"][2] +
                     timing["sim lsetup"][2] +
                     timing["sim lsolve"][2]))

    elif minmax == "minvar":
        # largest min time, smallest max time: least variability
        time.append(timing["sim"][1] -           # min = min - min
                    (timing["sim slow RHS"][1] +
                     timing["sim fast RHS"][1] +
                     timing["sim fast Jac"][1] +
                     timing["sim lsetup"][1] +
                     timing["sim lsolve"][1]))
        time.append(timing["sim"][2] -           # max = max - max
                    (timing["sim slow RHS"][2] +
                     timing["sim fast RHS"][2] +
                     timing["sim fast Jac"][2] +
                     timing["sim lsetup"][2] +
                     timing["sim lsolve"][2]))

    elif minmax == "maxvar":
        # smallest min time, largest max time: most variability
        time.append(timing["sim"][1] -           # min = min - max
                    (timing["sim slow RHS"][2] +
                     timing["sim fast RHS"][2] +
                     timing["sim fast Jac"][2] +
                     timing["sim lsetup"][2] +
                     timing["sim lsolve"][2]))
        time.append(timing["sim"][2] -           # max = max - min
                    (timing["sim slow RHS"][1] +
                     timing["sim fast RHS"][1] +
                     timing["sim fast Jac"][1] +
                     timing["sim lsetup"][1] +
                     timing["sim lsolve"][1]))
    else:
        raise Exception('unknown minmax option: skewmax, skewmin, minvar, maxvar')

    # check if min is larger than avg
    if time[1] > time[0]:
        if wminmax:
            print "Warning: sim sundials min > avg for file:",filename
            print "min = ",time[1]
            print "avg = ",time[0]
            print "Setting min = avg"
        time[1] = time[0]
    # check if max is smaller than avg
    if time[2] < time[0]:
        if wminmax:
            print "Warning: sim sundials max < avg for file:",filename
            print "max = ",time[2]
            print "avg = ",time[0]
            print "Setting max = avg"
        time[2] = time[0]

    timing["sim sundials"] = time

    # compute total times (trans + sim)
    keys = ["I/O", "MPI", "pack", "flux", "euler", "slow RHS", "fast RHS",
            "fast Jac", "lsetup", "lsolve", "sundials"]
    for k in keys:
        time = []
        for i in range(0,3):
            time.append(timing["trans "+k][i] + timing["sim "+k][i])
        timing["total "+k] = time

    # attaching timing dictonary
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

def plotscaling(rundata, keys, filterkey = [None, None],
                normalize = False, errorbars = False, title = None,
                Labels = None, save = False, fname = None, lwidth = 1):
    """
    Plot timing data as function of the number of processes

    Parameters:
    rundata:     list of test dictionaries to plot data from
    keys:        list of keys in timing dictionary of each test to plot
    filterkey:   filter test dictionaries, [ key to check, value to pass check ]
    normalize:   if True, normalize times by min value of first key in keys input
    title:       string to use for plot title
    Labels:      strings to use in legend for each plot
    save:        if True, save plot to file else show plot on screen
    lwidth:      line width to use in plots
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

    color = ['#e41a1c','#377eb8','#4daf4a','#984ea3',
             '#ff7f00','#a65628','#f781bf','#999999','#000000']

    # normalize based on the first key
    firstkey = True

    # loop counter
    i = 0

    # create figure and get axes
    fig = plt.figure()
    ax  = plt.axes()

    # get timing data for each key
    for k in keys:

        # clear list for timing data
        nprocs  = []
        time    = []
        mintime = []
        maxtime = []

        # extract timing data for each test
        for d in rundata:

            # check if tests are included/excluded based on a key
            if filterkey[0]:
                if d[filterkey[0]] == filterkey[1]:
                    nprocs.append(d["nprocs"][0])
                    time.append(d["timing"][k][0])
                    mintime.append(d["timing"][k][0] - d["timing"][k][1])
                    maxtime.append(d["timing"][k][2] - d["timing"][k][0])
            else:
                nprocs.append(d["nprocs"][0])
                time.append(d["timing"][k][0])
                mintime.append(d["timing"][k][0] - d["timing"][k][1])
                maxtime.append(d["timing"][k][2] - d["timing"][k][0])

        # convert lists to numpy arrays
        nprocs = np.array(nprocs)
        time   = np.array(time)
        minmax = np.array([mintime, maxtime])

        # normalize run times
        if normalize:
            if firstkey:
                reftime  = np.amin(time)
                firstkey = False
            time = time / reftime
            minmax = minmax / reftime

        if Labels:
            label_ = Labels[i]
        else:
            label_ = k

        # plot times
        if errorbars:
            plt.errorbar(nprocs, time, minmax,
                         color=color[i], label=label_, linewidth=lwidth)
        else:
            plt.semilogx(nprocs, time,
                         color=color[i], label=label_, linewidth=lwidth)

        # update counter
        i += 1

    # add title, labels, legend, etc.
    if title:
        plt.title(title)
    plt.xlabel("number of MPI tasks")
    if normalize:
        plt.ylabel("normalized run time (ref time = "+str(reftime)+" s)")
    else:
        plt.ylabel("run time (s)")

    if errorbars:
        ax.set_xscale('log')

    ax.set_yscale('log')


    # put the legend to the right of the current axis
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # remove error bars from legend
    handles, labels = ax.get_legend_handles_labels()
    if errorbars:
        handles = [h[0] for h in handles]

    # add legend
    lgd = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))

    # add background grid
    plt.grid(linestyle="--", alpha=0.2)

    # save or show figure
    if save:
        if fname:
            plt.savefig(fname, bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            plt.savefig("scaling.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ===============================================================================

def plotcomparescaling(rundata, keys, filterkey,
                       normalize = False, errorbars = False,
                       title = None, labels = None, save = False, fname = None,
                       lwidth = 1):
    """
    Compare timing data as function of the number of processes for two setups

    Parameters:
    rundata:     list of test dictionaries to plot data from
    keys:        list of keys in timing dictionary of each test to plot
    filterkey:   filter defining groups to compare, [ key to check, value for group 1 ]
    normalize:   if True, normalize times by min value of first key in keys input
    title:       string to use for plot title
    save:        if True, save plot to file else show plot on screen
    lwidth:      line width to use in plots
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

    color = ['#e41a1c','#377eb8','#4daf4a','#984ea3',
             '#ff7f00','#a65628','#f781bf','#999999','#000000']
    # color = ['#a6cee3','#1f78b4','#b2df8a','#33a02c',
    #          '#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6',
    #          '#6a3d9a','#ffff99','#b15928','#000000']

    # normalize based on the first key
    firstkey = True

    # loop counter
    i = 0

    # create figure and get axes
    fig = plt.figure()
    ax  = plt.axes()

    # get timing data for each key
    for k in keys:

        # clear list for timing data
        nprocs1  = []
        time1    = []
        mintime1 = []
        maxtime1 = []

        nprocs2  = []
        time2    = []
        mintime2 = []
        maxtime2 = []

        # extract timing data for each test
        for d in rundata:

            # check if tests are included/excluded based on a key
            if d[filterkey[0]] == filterkey[1]:
                nprocs1.append(d["nprocs"][0])
                time1.append(d["timing"][k][0])
                mintime1.append(d["timing"][k][0] - d["timing"][k][1])
                maxtime1.append(d["timing"][k][2] - d["timing"][k][0])
            else:
                nprocs2.append(d["nprocs"][0])
                time2.append(d["timing"][k][0])
                mintime2.append(d["timing"][k][0] - d["timing"][k][1])
                maxtime2.append(d["timing"][k][2] - d["timing"][k][0])

        # convert lists to numpy arrays
        nprocs1 = np.array(nprocs1)
        time1   = np.array(time1)
        minmax1 = np.array([mintime1, maxtime1])

        nprocs2 = np.array(nprocs2)
        time2   = np.array(time2)
        minmax2 = np.array([mintime2, maxtime2])

        # normalize run times
        if normalize:
            if firstkey:
                reftime = min(np.amin(time1), np.amin(time2))
                firstkey = False
            time1   = time1 / reftime
            minmax1 = minmax1 / reftime
            time2   = time2 / reftime
            minmax2 = minmax2 / reftime

        if labels:
            label1 = labels[i]
            label2 = labels[i+1]
        else:
            label1 = k
            label2 = k

        # plot times
        if errorbars:
            plt.errorbar(nprocs1, time1, minmax1,
                         color=color[i], linestyle='-', label=label1, linewidth=lwidth)
            plt.errorbar(nprocs2, time2, minmax2,
                         color=color[i+1], linestyle='--', label=label2, linewidth=lwidth)
        else:
            plt.semilogx(nprocs1, time1,
                         color=color[i], linestyle='-', label=label1, linewidth=lwidth)
            plt.semilogx(nprocs2, time2,
                         color=color[i+1], linestyle='--', label=label2, linewidth=lwidth)

        # update counter
        i += 2

    # add title, labels, legend, etc.
    if title:
        plt.title(title)
    plt.xlabel("number of MPI tasks")
    if normalize:
        plt.ylabel("normalized run time (ref time = "+str(reftime)+" s)")
    else:
        plt.ylabel("run time (s)")

    if errorbars:
        ax.set_xscale('log')

    # put the legend to the right of the current axis
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # remove error bars from legend
    handles, labels = ax.get_legend_handles_labels()
    if errorbars:
        handles = [h[0] for h in handles]

    # add legend
    lgd = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))

    # add background grid
    plt.grid(linestyle="--", alpha=0.2)

    # save or show figure
    if save:
        if fname:
            plt.savefig(fname, bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            plt.savefig("scaling.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ===============================================================================

def printpercenttime(rundata, keytotal, keys, filterkey,
                     normalize = False, errorbars = False):
    """
    Compare timing data as function of the number of processes for two setups

    Parameters:
    rundata:     list of test dictionaries to plot data from
    keys:        list of keys in timing dictionary of each test to plot
    filterkey:   filter defining groups to compare, [ key to check, value for group 1 ]
    normalize:   if True, normalize times by min value of first key in keys input
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

    # normalize based on the first key
    firstkey = True

    # loop counter
    i = 0

    # create figure and get axes
    fig = plt.figure()
    ax  = plt.axes()

    # get timing data for each key
    for k in keys:

        # clear list for timing data
        nprocs1  = []
        time1    = []
        mintime1 = []
        maxtime1 = []
        ptime1   = []
        minpvar1 = []
        maxpvar1 = []

        nprocs2  = []
        time2    = []
        mintime2 = []
        maxtime2 = []
        ptime2   = []
        minpvar2 = []
        maxpvar2 = []

        # extract timing data for each test
        for d in rundata:

            # check if tests are included/excluded based on a key
            if d[filterkey[0]] == filterkey[1]:
                nprocs1.append(d["nprocs"][0])
                time1.append(d["timing"][k][0])
                mintime1.append(d["timing"][k][1])
                maxtime1.append(d["timing"][k][2])
                ptime1.append(d["timing"][k][0] / d["timing"][keytotal][0] * 100)
                minpvar1.append((d["timing"][k][0] - d["timing"][k][1]) / d["timing"][k][0] * 100)
                maxpvar1.append((d["timing"][k][2] - d["timing"][k][0]) / d["timing"][k][0] * 100)
            else:
                nprocs2.append(d["nprocs"][0])
                time2.append(d["timing"][k][0])
                mintime2.append(d["timing"][k][1])
                maxtime2.append(d["timing"][k][2])
                ptime2.append(d["timing"][k][0] / d["timing"][keytotal][0] * 100)
                minpvar2.append((d["timing"][k][0] - d["timing"][k][1]) / d["timing"][k][0] * 100)
                maxpvar2.append((d["timing"][k][2] - d["timing"][k][0]) / d["timing"][k][0] * 100)

        # convert lists to numpy arrays
        nprocs1  = np.array(nprocs1)
        time1    = np.array(time1)
        mintime1 = np.array(mintime1)
        maxtime1 = np.array(maxtime1)
        ptime1   = np.array(ptime1)
        minpvar1 = np.array(minpvar1)
        maxpvar1 = np.array(maxpvar1)

        nprocs2  = np.array(nprocs2)
        time2    = np.array(time2)
        mintime2 = np.array(mintime2)
        maxtime2 = np.array(maxtime2)
        ptime2   = np.array(ptime2)
        minpvar2 = np.array(minpvar2)
        maxpvar2 = np.array(maxpvar2)

        # normalize run times
        if normalize:
            if firstkey:
                reftime  = min(np.amin(time1), np.amin(time2))
                firstkey = False
            ntime1 = time1 / reftime
            ntime2 = time2 / reftime

        formatstr1 = ("nprocs: {0:7d}, " +
                     "avg time: {1:8.2f}, " +
                     "avg normalized: {2:6.2f}, " +
                     "percent total: {3:6.2f}%")

        formatstr2 = ("nprocs: {0:7d}, " +
                      "avg time: {1:8.2f}, " +
                      "min time: {2:8.2f}, " +
                      "percent min var: {3:6.2f}%")

        formatstr3 = ("nprocs: {0:7d}, " +
                      "avg time: {1:8.2f}, " +
                      "max time: {2:8.2f}, " +
                      "percent max var: {3:6.2f}%")

        print k, "1"
        for w, x, y, z in np.nditer([nprocs1, time1, ntime1, ptime1]):
            print(formatstr1.format(int(w), float(x), float(y), float(z)))
        print

        print k, "1"
        for w, x, y, z in np.nditer([nprocs1, time1, mintime1, minpvar1]):
            print(formatstr2.format(int(w), float(x), float(y), float(z)))
        print

        print k, "1"
        for w, x, y, z in np.nditer([nprocs1, time1, maxtime1, maxpvar1]):
            print(formatstr3.format(int(w), float(x), float(y), float(z)))
        print

        print k, "2"
        for w, x, y, z in np.nditer([nprocs2, time2, ntime2, ptime2]):
            print(formatstr1.format(int(w), float(x), float(y), float(z)))
        print

        print k, "2"
        for w, x, y, z in np.nditer([nprocs2, time2, mintime2, minpvar2]):
            print(formatstr2.format(int(w), float(x), float(y), float(z)))
        print

        print k, "2"
        for w, x, y, z in np.nditer([nprocs2, time2, maxtime2, maxpvar2]):
            print(formatstr3.format(int(w), float(x), float(y), float(z)))
        print

# ===============================================================================

def sortnprocs(elem):
    return elem["nprocs"][0]

# ===============================================================================

if __name__ == "__main__":
    main()

# EOF
