#!/usr/bin/env python3
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ SMU
#------------------------------------------------------------
# Copyright (c) 2019, Southern Methodist University.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------
# matplotlib-based plotting utility function for
# z-directional SOD shock tube problem

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
nx, ny, nz, nt, xgrid, ygrid, zgrid, tgrid, rho, mx, my, mz, et = load_data()

# set gas constant
gamma = 1.4

# output general information to screen
print('Generating plots for data set:')
print('  nz: ', nz)
print('  nt: ', nt)
    
# determine extents of plots
minmaxrho = [0.9*rho.min(), 1.1*rho.max()]
if (rho.min() == rho.max()):
    minmaxrho = [rho.min()-0.1, rho.max()+0.1]
minmaxmz  = [0.9*mz.min(), 1.1*mz.max()]
if (mz.min() == mz.max()):
    minmaxmz = [mz.min()-0.1, mz.max()+0.1]
minmaxet  = [0.9*et.min(), 1.1*et.max()]
if (et.min() == et.max()):
    minmaxet = [et.min()-0.1, et.max()+0.1]

# generate plots of solution
for tstep in range(nt):
    
    print('time step', tstep+1, 'out of', nt)
            
    # set string constants for current time, mesh sizes
    tstr  = repr(tstep)
    nzstr = repr(nz)

    rhoname = 'rho.' + repr(tstep).zfill(4) + '.png'
    mzname  = 'mz.'  + repr(tstep).zfill(4) + '.png'
    etname  = 'et.'  + repr(tstep).zfill(4) + '.png'

    # get exact solutions at this time
    rhotrue, utrue, mztrue, ptrue, ettrue = exact_Riemann(tgrid[tstep], zgrid, 0.5, gamma, 
                                                          1.0, 0.125, 0.0, 0.0, 1.0, 0.1)
    
    # plot line graphs of current solution and save to disk
    plt.figure(1)
    plt.plot(zgrid, rho[nx//2,ny//2,:,tstep], 'b--', zgrid, rhotrue, 'k-')
    plt.legend(('computed','true'))
    plt.xlabel('z')
    plt.ylabel(r'$\rho$')
    plt.ylim((minmaxrho[0], minmaxrho[1]))
    plt.title(r'$\rho($' + tstr + r'$,z)$, mesh = ' + nzstr)
    plt.savefig(rhoname)
    
    plt.figure(2)
    plt.plot(zgrid, mz[nx//2,ny//2,:,tstep], 'b--', zgrid, mztrue, 'k-')
    plt.legend(('computed','true'))
    plt.xlabel('z')
    plt.ylabel(r'$m_z$')
    plt.ylim((minmaxmz[0], minmaxmz[1]))
    plt.title(r'$m_z($' + tstr + r'$,z)$, mesh = ' + nzstr)
    plt.savefig(mzname)
    
    plt.figure(3)
    plt.plot(zgrid, et[nx//2,ny//2,:,tstep], 'b--', zgrid, ettrue, 'k-')
    plt.legend(('computed','true'))
    plt.xlabel('z')
    plt.ylabel(r'$e_t$')
    plt.ylim((minmaxet[0], minmaxet[1]))
    plt.title(r'$e_t($' + tstr + r'$,z)$, mesh = ' + nzstr)
    plt.savefig(etname)

    if (showplots):
        plt.show()
    plt.figure(1), plt.close()
    plt.figure(2), plt.close()
    plt.figure(3), plt.close()
        
        
  
##### end of script #####
