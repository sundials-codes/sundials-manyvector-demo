#!/usr/bin/env python3
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ SMU
#------------------------------------------------------------
# Copyright (c) 2019, Southern Methodist University.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------
# matplotlib-based plotting utility function for
# hurricane test problem in the yz-plane

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
yl    = -1.0
yr    =  1.0
zl    = -1.0
zr    =  1.0


# utility function to create analytical solution
def analytical_solution(t,ny,nz):
  if (t == 0):
    t = 1e-14
  p0prime = Amp*gamma*rho0**(gamma-1.0)
  rthresh = 2.0*t*np.sqrt(p0prime)
  rho = np.zeros((ny,nz), dtype=float)
  my  = np.zeros((ny,nz), dtype=float)
  mz  = np.zeros((ny,nz), dtype=float)
  dy  = (yr-yl)/ny
  dz  = (zr-zl)/nz
  for j in range(nz):
    for i in range(ny):
      y = (i+0.5)*dy + yl
      z = (j+0.5)*dz + zl
      r = np.sqrt(y*y + z*z)
      if (r == 0.0):  # protect against division by zero
        r = 1e-14  
      costheta = y/r
      sintheta = z/r
      if (r < rthresh):
        rho[i,j] = r*r / (8*Amp*t*t)
        my[i,j] = rho[i,j] * (y + z) / (2*t)
        mz[i,j] = rho[i,j] * (z - y) / (2*t)
      else:
        rho[i,j] = rho0
        my[i,j] = rho0 * ( 2*t*p0prime*costheta +
                           np.sqrt(2*p0prime)*np.sqrt(r*r-2*t*t*p0prime)*sintheta )/r
        mz[i,j] = rho0 * ( 2*t*p0prime*sintheta -
                           np.sqrt(2*p0prime)*np.sqrt(r*r-2*t*t*p0prime)*costheta )/r
  return [rho, my, mz]


# load solution data
nx, ny, nz, nt, xgrid, ygrid, zgrid, tgrid, rho, mx, my, mz, et = load_data()

# output general information to screen
print('Generating plots for data set:')
print('  ny: ', ny)
print('  nz: ', nz)
print('  nt: ', nt)
    
# determine extents of plots
minmaxrho = [0.9*rho.min(), 1.1*rho.max()]
if (rho.min() == rho.max()):
    minmaxrho = [rho.min()-0.1, rho.max()+0.1]
minmaxmy  = [0.9*my.min(), 1.1*my.max()]
if (my.min() == my.max()):
    minmaxmy = [my.min()-0.1, my.max()+0.1]
minmaxmz  = [0.9*mz.min(), 1.1*mz.max()]
if (mz.min() == mz.max()):
    minmaxmz = [mz.min()-0.1, mz.max()+0.1]
minmaxet  = [0.9*et.min(), 1.1*et.max()]
if (et.min() == et.max()):
    minmaxet = [et.min()-0.1, et.max()+0.1]

# generate plots of solution
for tstep in range(nt):
    
    print('time step', tstep+1, 'out of', nt)

    # get true solutions
    rhotrue, mytrue, mztrue = analytical_solution(tgrid[tstep],ny,nz)
    
    # set string constants for current time, mesh sizes
    tstr  = repr(tstep)
    nystr = repr(ny)
    nzstr = repr(nz)

    # set filenames for graphics
    rhosurf = 'rho_surface.' + repr(tstep).zfill(4) + '.png'
    rhocont = 'rho_contour.' + repr(tstep).zfill(4) + '.png'
    rhotr   = 'rho_true.'    + repr(tstep).zfill(4) + '.png'
    mysurf  = 'my_surface.'  + repr(tstep).zfill(4) + '.png'
    mycont  = 'my_contour.'  + repr(tstep).zfill(4) + '.png'
    mytr    = 'my_true.'     + repr(tstep).zfill(4) + '.png'
    mzsurf  = 'mz_surface.'  + repr(tstep).zfill(4) + '.png'
    mzcont  = 'mz_contour.'  + repr(tstep).zfill(4) + '.png'
    mztr    = 'mz_true.'     + repr(tstep).zfill(4) + '.png'
    etsurf  = 'et_surface.'  + repr(tstep).zfill(4) + '.png'
    etcont  = 'et_contour.'  + repr(tstep).zfill(4) + '.png'

    # set y and z meshgrid objects
    Z,Y = np.meshgrid(zgrid,ygrid)

    # plot current solution as surfaces, and save to disk
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Z, Y, rho[nx//2,:,:,tstep], rstride=1, cstride=1, 
                    cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel('z'); ax.set_ylabel('y'); ax.set_zlim((minmaxrho[0], minmaxrho[1]))
    ax.view_init(elevation,angle);
    plt.title(r'$\rho(y,z)$ at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
    plt.savefig(rhosurf)
            
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Z, Y, my[nx//2,:,:,tstep], rstride=1, cstride=1, 
                    cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel('z'); ax.set_ylabel('y'); ax.set_zlim((minmaxmy[0], minmaxmy[1]))
    ax.view_init(elevation,angle);
    plt.title(r'$m_y(y,z)$ at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
    plt.savefig(mysurf)
            
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Z, Y, mz[nx//2,:,:,tstep], rstride=1, cstride=1, 
                    cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel('z'); ax.set_ylabel('y'); ax.set_zlim((minmaxmz[0], minmaxmz[1]))
    ax.view_init(elevation,angle);
    plt.title(r'$m_z(y,z)$ at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
    plt.savefig(mzsurf)
            
    fig = plt.figure(4)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Z, Y, et[nx//2,:,:,tstep], rstride=1, cstride=1, 
                    cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel('z'); ax.set_ylabel('y'); ax.set_zlim((minmaxet[0], minmaxet[1]))
    ax.view_init(elevation,angle);
    plt.title(r'$e_t(y,z)$ at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
    plt.savefig(etsurf)
    
    # plot current solution as contours, and save to disk
    fig = plt.figure(5)
    plt.contourf(Z, Y, rho[nx//2,:,:,tstep])
    plt.colorbar();  plt.xlabel('z'); plt.ylabel('y')
    plt.title(r'$\rho(y,z)$ at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
    plt.savefig(rhocont)
            
    fig = plt.figure(6)
    plt.contourf(Z, Y, my[nx//2,:,:,tstep])
    plt.colorbar();  plt.xlabel('z'); plt.ylabel('y')
    plt.title(r'$m_y(y,z)$ at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
    plt.savefig(mycont)
            
    fig = plt.figure(7)
    plt.contourf(Z, Y, mz[nx//2,:,:,tstep])
    plt.colorbar();  plt.xlabel('z'); plt.ylabel('y')
    plt.title(r'$m_z(y,z)$ at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
    plt.savefig(mzcont)
            
    fig = plt.figure(8)
    plt.contourf(Z, Y, et[nx//2,:,:,tstep])
    plt.colorbar();  plt.xlabel('z'); plt.ylabel('y')
    plt.title(r'$e_t(y,z)$ at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
    plt.savefig(etcont)
    
    # plot true solution as contours, and save to disk
    fig = plt.figure(9)
    plt.contourf(Z, Y, rhotrue)
    plt.colorbar();  plt.xlabel('z'); plt.ylabel('y')
    plt.title(r'$\rho_{true}(y,z)$ at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
    plt.savefig(rhotr)
            
    fig = plt.figure(10)
    plt.contourf(Z, Y, mytrue)
    plt.colorbar();  plt.xlabel('z'); plt.ylabel('y')
    plt.title(r'$m_{y,true}(y,z)$ at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
    plt.savefig(mytr)
            
    fig = plt.figure(11)
    plt.contourf(Z, Y, mztrue)
    plt.colorbar();  plt.xlabel('z'); plt.ylabel('y')
    plt.title(r'$m_{z,true}(y,z)$ at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
    plt.savefig(mztr)
            
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
