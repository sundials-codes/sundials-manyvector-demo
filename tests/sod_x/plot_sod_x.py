#!/usr/bin/env python3
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ SMU
#------------------------------------------------------------
# Copyright (c) 2019, Southern Methodist University.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------
# matplotlib-based plotting utility function for
# x-directional SOD shock tube problem

# imports
import numpy as np
import matplotlib.pyplot as plt
from utilities_euler3D import *


# determine if running interactively
if __name__=="__main__":
  showplots = False
else:
  showplots = True


# load solution data
nx, ny, nz, nchem, nt, xgrid, ygrid, zgrid, tgrid, rho, mx, my, mz, et, chem = load_data()

# set gas constant
gamma = 1.4

# output general information to screen
print('Generating plots for data set:')
print('  nx: ', nx)
print('  nt: ', nt)
    
# determine extents of plots
minmaxrho = [0.9*rho.min(), 1.1*rho.max()]
if (rho.min() == rho.max()):
    minmaxrho = [rho.min()-0.1, rho.max()+0.1]
minmaxmx  = [0.9*mx.min(), 1.1*mx.max()]
if (mx.min() == mx.max()):
    minmaxmx = [mx.min()-0.1, mx.max()+0.1]
minmaxet  = [0.9*et.min(), 1.1*et.max()]
if (et.min() == et.max()):
    minmaxet = [et.min()-0.1, et.max()+0.1]

# generate plots of solution
for tstep in range(nt):
    
    print('time step', tstep+1, 'out of', nt)
            
    # set string constants for current time, mesh sizes
    tstr  = repr(tstep)
    nxstr = repr(nx)

    rhoname = 'rho.' + repr(tstep).zfill(4) + '.png'
    mxname  = 'mx.'  + repr(tstep).zfill(4) + '.png'
    etname  = 'et.'  + repr(tstep).zfill(4) + '.png'

    # get exact solutions at this time
    rhotrue, utrue, mxtrue, ptrue, ettrue = exact_Riemann(tgrid[tstep], xgrid, 0.5, gamma, 
                                                          1.0, 0.125, 0.0, 0.0, 1.0, 0.1)
    
    # plot line graphs of current solution and save to disk
    plt.figure(1)
    plt.plot(xgrid, rho[:,ny//2,nz//2,tstep], 'b--', xgrid, rhotrue, 'k-')
    plt.legend(('computed','true'))
    plt.xlabel('x')
    plt.ylabel(r'$\rho$')
    plt.ylim((minmaxrho[0], minmaxrho[1]))
    plt.title(r'$\rho($' + tstr + r'$,x)$, mesh = ' + nxstr)
    plt.savefig(rhoname)
    
    plt.figure(2)
    plt.plot(xgrid, mx[:,ny//2,nz//2,tstep], 'b--', xgrid, mxtrue, 'k-')
    plt.legend(('computed','true'))
    plt.xlabel('x')
    plt.ylabel(r'$m_x$')
    plt.ylim((minmaxmx[0], minmaxmx[1]))
    plt.title(r'$m_x($' + tstr + r'$,x)$, mesh = ' + nxstr)
    plt.savefig(mxname)
    
    plt.figure(3)
    plt.plot(xgrid, et[:,ny//2,nz//2,tstep], 'b--', xgrid, ettrue, 'k-')
    plt.legend(('computed','true'))
    plt.xlabel('x')
    plt.ylabel(r'$e_t$')
    plt.ylim((minmaxet[0], minmaxet[1]))
    plt.title(r'$e_t($' + tstr + r'$,x)$, mesh = ' + nxstr)
    plt.savefig(etname)

    if (showplots):
        plt.show()
    plt.figure(1), plt.close()
    plt.figure(2), plt.close()
    plt.figure(3), plt.close()
        
        
  
##### end of script #####
