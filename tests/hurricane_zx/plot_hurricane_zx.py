#!/usr/bin/env python3
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ SMU
#------------------------------------------------------------
# Copyright (c) 2019, Southern Methodist University.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------
# matplotlib-based plotting utility function for
# hurricane test problem in the zx-plane

# imports
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from utilities_euler3D import *


# determine if running interactively
if __name__=="__main__":
  showplots = False
else:
  showplots = True

# set view for surface plots  
elevation = 15
angle = 20


# set test constants
rho0  = 1.0
v0    = 10.0
Amp   = 25.0
gamma = 2.0
zl    = -1.0
zr    =  1.0
xl    = -1.0
xr    =  1.0


# utility function to create analytical solution
def analytical_solution(t,nz,nx):
  if (t == 0):
    t = 1e-14
  p0prime = Amp*gamma*rho0**(gamma-1.0)
  rthresh = 2.0*t*np.sqrt(p0prime)
  rho = np.zeros((nz,nx), dtype=float)
  mz  = np.zeros((nz,nx), dtype=float)
  mx  = np.zeros((nz,nx), dtype=float)
  dz  = (zr-zl)/nz
  dx  = (xr-xl)/nx
  for j in range(nx):
    for i in range(nz):
      z = (i+0.5)*dz + zl
      x = (j+0.5)*dx + xl
      r = np.sqrt(z*z + x*x)
      if (r == 0.0):  # protect against division by zero
        r = 1e-14  
      costheta = z/r
      sintheta = x/r
      if (r < rthresh):
        rho[i,j] = r*r / (8*Amp*t*t)
        mz[i,j] = rho[i,j] * (z + x) / (2*t)
        mx[i,j] = rho[i,j] * (x - z) / (2*t)
      else:
        rho[i,j] = rho0
        mz[i,j] = rho0 * ( 2*t*p0prime*costheta +
                           np.sqrt(2*p0prime)*np.sqrt(r*r-2*t*t*p0prime)*sintheta )/r
        mx[i,j] = rho0 * ( 2*t*p0prime*sintheta -
                           np.sqrt(2*p0prime)*np.sqrt(r*r-2*t*t*p0prime)*costheta )/r
  return [rho, mz, mx]


# load solution data
nx, ny, nz, nt, xgrid, ygrid, zgrid, tgrid, rho, mx, my, mz, et = load_data()

# output general information to screen
print('Generating plots for data set:')
print('  nz: ', nz)
print('  nx: ', nx)
print('  nt: ', nt)
    
# determine extents of plots
minmaxrho = [0.9*rho.min(), 1.1*rho.max()]
if (rho.min() == rho.max()):
    minmaxrho = [rho.min()-0.1, rho.max()+0.1]
minmaxmz  = [0.9*mz.min(), 1.1*mz.max()]
if (mz.min() == mz.max()):
    minmaxmz = [mz.min()-0.1, mz.max()+0.1]
minmaxmx  = [0.9*mx.min(), 1.1*mx.max()]
if (mx.min() == mx.max()):
    minmaxmx = [mx.min()-0.1, mx.max()+0.1]
minmaxet  = [0.9*et.min(), 1.1*et.max()]
if (et.min() == et.max()):
    minmaxet = [et.min()-0.1, et.max()+0.1]

# generate plots of solution
for tstep in range(nt):
    
    print('time step', tstep+1, 'out of', nt)

    # get true solutions
    rhotrue, mztrue, mxtrue = analytical_solution(tgrid[tstep],nz,nx)
    
    # set string constants for current time, mesh sizes
    tstr  = repr(tstep)
    nzstr = repr(nz)
    nxstr = repr(nx)

    # set filenames for graphics
    rhosurf = 'rho_surface.' + repr(tstep).zfill(4) + '.png'
    rhocont = 'rho_contour.' + repr(tstep).zfill(4) + '.png'
    rhotr   = 'rho_true.'    + repr(tstep).zfill(4) + '.png'
    mzsurf  = 'mz_surface.'  + repr(tstep).zfill(4) + '.png'
    mzcont  = 'mz_contour.'  + repr(tstep).zfill(4) + '.png'
    mztr    = 'mz_true.'     + repr(tstep).zfill(4) + '.png'
    mxsurf  = 'mx_surface.'  + repr(tstep).zfill(4) + '.png'
    mxcont  = 'mx_contour.'  + repr(tstep).zfill(4) + '.png'
    mxtr    = 'mx_true.'     + repr(tstep).zfill(4) + '.png'
    etsurf  = 'et_surface.'  + repr(tstep).zfill(4) + '.png'
    etcont  = 'et_contour.'  + repr(tstep).zfill(4) + '.png'

    # set z and x meshgrid objects
    X,Z = np.meshgrid(xgrid,zgrid)

    # plot current solution as surfaces, and save to disk
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Z, X, rho[:,ny//2,:,tstep], rstride=1, cstride=1, 
                    cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel('z'); ax.set_ylabel('x'); ax.set_zlim((minmaxrho[0], minmaxrho[1]))
    ax.view_init(elevation,angle);
    plt.title(r'$\rho(x,z)$ at output ' + tstr + ', mesh = ' + nxstr + 'z' + nzstr)
    plt.savefig(rhosurf)
            
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Z, X, mz[:,ny//2,:,tstep], rstride=1, cstride=1, 
                    cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel('z'); ax.set_ylabel('x'); ax.set_zlim((minmaxmz[0], minmaxmz[1]))
    ax.view_init(elevation,angle);
    plt.title(r'$m_z(x,z)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
    plt.savefig(mzsurf)
            
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Z, X, mx[:,ny//2,:,tstep], rstride=1, cstride=1, 
                    cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel('z'); ax.set_ylabel('x'); ax.set_zlim((minmaxmx[0], minmaxmx[1]))
    ax.view_init(elevation,angle);
    plt.title(r'$m_x(x,z)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
    plt.savefig(mxsurf)
            
    fig = plt.figure(4)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Z, X, et[:,ny//2,:,tstep], rstride=1, cstride=1, 
                    cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel('z'); ax.set_ylabel('x'); ax.set_zlim((minmaxet[0], minmaxet[1]))
    ax.view_init(elevation,angle);
    plt.title(r'$e_t(x,z)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
    plt.savefig(etsurf)
    
    # plot current solution as contours, and save to disk
    fig = plt.figure(5)
    plt.contourf(Z, X, rho[:,ny//2,:,tstep])
    plt.colorbar();  plt.xlabel('z'); plt.ylabel('x')
    plt.title(r'$\rho(x,z)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
    plt.savefig(rhocont)
            
    fig = plt.figure(6)
    plt.contourf(Z, X, mz[:,ny//2,:,tstep])
    plt.colorbar();  plt.xlabel('z'); plt.ylabel('x')
    plt.title(r'$m_z(x,z)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
    plt.savefig(mzcont)
            
    fig = plt.figure(7)
    plt.contourf(Z, X, mx[:,ny//2,:,tstep])
    plt.colorbar();  plt.xlabel('z'); plt.ylabel('x')
    plt.title(r'$m_x(x,z)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
    plt.savefig(mxcont)
            
    fig = plt.figure(8)
    plt.contourf(Z, X, et[:,ny//2,:,tstep])
    plt.colorbar();  plt.xlabel('z'); plt.ylabel('x')
    plt.title(r'$e_t(x,z)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
    plt.savefig(etcont)
    
    # plot true solution as contours, and save to disk
    fig = plt.figure(9)
    plt.contourf(Z, X, np.transpose(rhotrue))
    plt.colorbar();  plt.xlabel('z'); plt.ylabel('x')
    plt.title(r'$\rho_{true}(x,z)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
    plt.savefig(rhotr)
            
    fig = plt.figure(10)
    plt.contourf(Z, X, np.transpose(mztrue))
    plt.colorbar();  plt.xlabel('z'); plt.ylabel('x')
    plt.title(r'$m_{z,true}(x,z)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
    plt.savefig(mztr)
            
    fig = plt.figure(11)
    plt.contourf(Z, X, np.transpose(mxtrue))
    plt.colorbar();  plt.xlabel('z'); plt.ylabel('x')
    plt.title(r'$m_{x,true}(x,z)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
    plt.savefig(mxtr)
            
    if (showplots):
      plt.show()
    plt.figure(1), plt.close()
    plt.figure(2), plt.close()
    plt.figure(3), plt.close()
    plt.figure(4), plt.close()
    plt.figure(5), plt.close()
    plt.figure(6), plt.close()
    plt.figure(7), plt.close()
    plt.figure(8), plt.close()

    plt.figure(9),  plt.close()
    plt.figure(10), plt.close()
    plt.figure(11), plt.close()
    
        
  
##### end of script #####
