#!/usr/bin/env python3
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ SMU
#------------------------------------------------------------
# Copyright (c) 2019, Southern Methodist University.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------
# matplotlib-based plotting utility function for
# y-directional SOD shock tube problem

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

# set gas constant, problem setup constants
gamma = 1.4
rho0 = 1.0
p0 = 1.0
Amp = 0.1
v0 = 0.5

# output general information to screen
print('Generating plots for data set:')
print('  ny: ', ny)
print('  nt: ', nt)
    
# determine extents of plots
minmaxrho = [0.9*rho.min(), 1.1*rho.max()]
if (rho.min() == rho.max()):
    minmaxrho = [rho.min()-0.1, rho.max()+0.1]
minmaxmy  = [0.9*my.min(), 1.1*my.max()]
if (my.min() == my.max()):
    minmaxmy = [my.min()-0.1, my.max()+0.1]
minmaxet  = [0.9*et.min(), 1.1*et.max()]
if (et.min() == et.max()):
    minmaxet = [et.min()-0.1, et.max()+0.1]

# generate plots of solution
for tstep in range(nt):
    
    print('time step', tstep+1, 'out of', nt)
            
    # set string constants for current time, mesh sizes
    tstr  = repr(tstep)
    nystr = repr(ny)

    rhoname = 'rho.' + repr(tstep).zfill(4) + '.png'
    myname  = 'my.'  + repr(tstep).zfill(4) + '.png'
    etname  = 'et.'  + repr(tstep).zfill(4) + '.png'

    # compute exact solutions at this time
    rhotrue = rho0 + Amp*np.sin(2*np.pi*(ygrid-v0*tgrid[tstep]))
    mytrue = v0*rhotrue
    ettrue = p0/(gamma-1.0) + 0.5*v0*v0*rhotrue
    
    # plot line graphs of current solution and save to disk
    plt.figure(1)
    plt.plot(ygrid, rho[nx//2,:,nz//2,tstep], 'b--', ygrid, rhotrue, 'k-')
    plt.legend(('computed','true'))
    plt.xlabel('y')
    plt.ylabel(r'$\rho$')
    plt.ylim((minmaxrho[0], minmaxrho[1]))
    plt.title(r'$\rho($' + tstr + r'$,y)$, mesh = ' + nystr)
    plt.savefig(rhoname)
    
    plt.figure(2)
    plt.plot(ygrid, my[nx//2,:,nz//2,tstep], 'b--', ygrid, mytrue, 'k-')
    plt.legend(('computed','true'))
    plt.xlabel('y')
    plt.ylabel(r'$m_y$')
    plt.ylim((minmaxmy[0], minmaxmy[1]))
    plt.title(r'$m_y($' + tstr + r'$,y)$, mesh = ' + nystr)
    plt.savefig(myname)
    
    plt.figure(3)
    plt.plot(ygrid, et[nx//2,:,nz//2,tstep], 'b--', ygrid, ettrue, 'k-')
    plt.legend(('computed','true'))
    plt.xlabel('y')
    plt.ylabel(r'$e_t$')
    plt.ylim((minmaxet[0], minmaxet[1]))
    plt.title(r'$e_t($' + tstr + r'$,y)$, mesh = ' + nystr)
    plt.savefig(etname)

    if (showplots):
        plt.show()
    plt.figure(1), plt.close()
    plt.figure(2), plt.close()
    plt.figure(3), plt.close()
        
        
  
##### end of script #####
