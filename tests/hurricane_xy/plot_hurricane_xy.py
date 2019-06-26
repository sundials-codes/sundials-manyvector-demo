#!/usr/bin/env python3
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ SMU
#------------------------------------------------------------
# Copyright (c) 2019, Southern Methodist University.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------
# matplotlib-based plotting utility function for
# hurricane test problem in the xy-plane

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
xl    = -1.0
xr    =  1.0
yl    = -1.0
yr    =  1.0


# utility function to create analytical solution
def analytical_solution(t,nx,ny):
  if (t == 0):
    t = 1e-14
  p0prime = Amp*gamma*rho0**(gamma-1.0)
  rthresh = 2.0*t*np.sqrt(p0prime)
  rho = np.zeros((nx,ny), dtype=float)
  mx  = np.zeros((nx,ny), dtype=float)
  my  = np.zeros((nx,ny), dtype=float)
  dx  = (xr-xl)/nx
  dy  = (yr-yl)/ny
  for j in range(ny):
    for i in range(nx):
      x = (i+0.5)*dx + xl
      y = (j+0.5)*dy + yl
      r = np.sqrt(x*x + y*y)
      if (r == 0.0):  # protect against division by zero
        r = 1e-14  
      costheta = x/r
      sintheta = y/r
      if (r < rthresh):
        rho[i,j] = r*r / (8*Amp*t*t)
        mx[i,j] = rho[i,j] * (x + y) / (2*t)
        my[i,j] = rho[i,j] * (y - x) / (2*t)
      else:
        rho[i,j] = rho0
        mx[i,j] = rho0 * ( 2*t*p0prime*costheta +
                           np.sqrt(2*p0prime)*np.sqrt(r*r-2*t*t*p0prime)*sintheta )/r
        my[i,j] = rho0 * ( 2*t*p0prime*sintheta -
                           np.sqrt(2*p0prime)*np.sqrt(r*r-2*t*t*p0prime)*costheta )/r
  return [rho, mx, my]


# load solution data
nx, ny, nz, nt, xgrid, ygrid, zgrid, tgrid, rho, mx, my, mz, et = load_data()

# output general information to screen
print('Generating plots for data set:')
print('  nx: ', nx)
print('  ny: ', ny)
print('  nt: ', nt)
    
# determine extents of plots
minmaxrho = [0.9*rho.min(), 1.1*rho.max()]
if (rho.min() == rho.max()):
    minmaxrho = [rho.min()-0.1, rho.max()+0.1]
minmaxmx  = [0.9*mx.min(), 1.1*mx.max()]
if (mx.min() == mx.max()):
    minmaxmx = [mx.min()-0.1, mx.max()+0.1]
minmaxmy  = [0.9*my.min(), 1.1*my.max()]
if (my.min() == my.max()):
    minmaxmy = [my.min()-0.1, my.max()+0.1]
minmaxet  = [0.9*et.min(), 1.1*et.max()]
if (et.min() == et.max()):
    minmaxet = [et.min()-0.1, et.max()+0.1]

# generate plots of solution
for tstep in range(nt):
    
    print('time step', tstep+1, 'out of', nt)

    # get true solutions
    rhotrue, mxtrue, mytrue = analytical_solution(tgrid[tstep],nx,ny)
    
    # set string constants for current time, mesh sizes
    tstr  = repr(tstep)
    nxstr = repr(nx)
    nystr = repr(ny)

    # set filenames for graphics
    rhosurf = 'rho_surface.' + repr(tstep).zfill(4) + '.png'
    rhocont = 'rho_contour.' + repr(tstep).zfill(4) + '.png'
    rhotr   = 'rho_true.'    + repr(tstep).zfill(4) + '.png'
    mxsurf  = 'mx_surface.'  + repr(tstep).zfill(4) + '.png'
    mxcont  = 'mx_contour.'  + repr(tstep).zfill(4) + '.png'
    mxtr    = 'mx_true.'     + repr(tstep).zfill(4) + '.png'
    mysurf  = 'my_surface.'  + repr(tstep).zfill(4) + '.png'
    mycont  = 'my_contour.'  + repr(tstep).zfill(4) + '.png'
    mytr    = 'my_true.'     + repr(tstep).zfill(4) + '.png'
    etsurf  = 'et_surface.'  + repr(tstep).zfill(4) + '.png'
    etcont  = 'et_contour.'  + repr(tstep).zfill(4) + '.png'

    # set x and z meshgrid objects
    Y,X = np.meshgrid(ygrid,xgrid)

    # plot current solution as surfaces, and save to disk
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Y, X, rho[:,:,nz//2,tstep], rstride=1, cstride=1, 
                    cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel('y'); ax.set_ylabel('x'); ax.set_zlim((minmaxrho[0], minmaxrho[1]))
    ax.view_init(elevation,angle);
    plt.title(r'$\rho(x,y)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
    plt.savefig(rhosurf)
            
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Y, X, mx[:,:,nz//2,tstep], rstride=1, cstride=1, 
                    cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel('y'); ax.set_ylabel('x'); ax.set_zlim((minmaxmx[0], minmaxmx[1]))
    ax.view_init(elevation,angle);
    plt.title(r'$m_x(x,y)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
    plt.savefig(mxsurf)
            
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Y, X, my[:,:,nz//2,tstep], rstride=1, cstride=1, 
                    cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel('y'); ax.set_ylabel('x'); ax.set_zlim((minmaxmy[0], minmaxmy[1]))
    ax.view_init(elevation,angle);
    plt.title(r'$m_y(x,y)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
    plt.savefig(mysurf)
            
    fig = plt.figure(4)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Y, X, et[:,:,nz//2,tstep], rstride=1, cstride=1, 
                    cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel('y'); ax.set_ylabel('x'); ax.set_zlim((minmaxet[0], minmaxet[1]))
    ax.view_init(elevation,angle);
    plt.title(r'$e_t(x,y)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
    plt.savefig(etsurf)
    
    # plot current solution as contours, and save to disk
    fig = plt.figure(5)
    plt.contourf(Y, X, rho[:,:,nz//2,tstep])
    plt.colorbar();  plt.xlabel('y'); plt.ylabel('x')
    plt.title(r'$\rho(x,y)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
    plt.savefig(rhocont)
            
    fig = plt.figure(6)
    plt.contourf(Y, X, mx[:,:,nz//2,tstep])
    plt.colorbar();  plt.xlabel('y'); plt.ylabel('x')
    plt.title(r'$m_x(x,y)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
    plt.savefig(mxcont)
            
    fig = plt.figure(7)
    plt.contourf(Y, X, my[:,:,nz//2,tstep])
    plt.colorbar();  plt.xlabel('y'); plt.ylabel('x')
    plt.title(r'$m_y(x,y)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
    plt.savefig(mycont)
            
    fig = plt.figure(8)
    plt.contourf(Y, X, et[:,:,nz//2,tstep])
    plt.colorbar();  plt.xlabel('y'); plt.ylabel('x')
    plt.title(r'$e_t(x,y)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
    plt.savefig(etcont)
    
    # plot true solution as contours, and save to disk
    fig = plt.figure(9)
    plt.contourf(Y, X, rhotrue)
    plt.colorbar();  plt.xlabel('y'); plt.ylabel('x')
    plt.title(r'$\rho_{true}(x,y)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
    plt.savefig(rhotr)
            
    fig = plt.figure(10)
    plt.contourf(Y, X, mxtrue)
    plt.colorbar();  plt.xlabel('y'); plt.ylabel('x')
    plt.title(r'$m_{x,true}(x,y)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
    plt.savefig(mxtr)
            
    fig = plt.figure(11)
    plt.contourf(Y, X, mytrue)
    plt.colorbar();  plt.xlabel('y'); plt.ylabel('x')
    plt.title(r'$m_{y,true}(x,y)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
    plt.savefig(mytr)
            
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
