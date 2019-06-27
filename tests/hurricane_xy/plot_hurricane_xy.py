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
    numfigs = 0
    
    print('time step', tstep+1, 'out of', nt)

    # get true solutions
    rhotrue, mxtrue, mytrue = analytical_solution(tgrid[tstep],nx,ny)
    
    # set string constants for current time, mesh sizes
    tstr  = repr(tstep)
    nxstr = repr(nx)
    nystr = repr(ny)

    # extract 2D velocity fields (computed and true)
    U = mx[:,:,nz//2,tstep]/rho[:,:,nz//2,tstep]
    Utrue = mxtrue/rhotrue
    V = my[:,:,nz//2,tstep]/rho[:,:,nz//2,tstep]
    Vtrue = mytrue/rhotrue
    speed = np.sqrt(U**2 + V**2)
    speedtrue = np.sqrt(Utrue**2 + Vtrue**2)
    
    # set filenames for graphics
    rhosurf  = 'rho_surface.'   + repr(tstep).zfill(4) + '.png'
    etsurf   = 'et_surface.'    + repr(tstep).zfill(4) + '.png'
    vstr     = 'velocity.'      + repr(tstep).zfill(4) + '.png'
    rhocont  = 'rho_contour.'   + repr(tstep).zfill(4) + '.png'
    etcont   = 'et_contour.'    + repr(tstep).zfill(4) + '.png'
    rho1dout = 'rho1d.'         + repr(tstep).zfill(4) + '.png'
    mx1dout  = 'mx1d.'          + repr(tstep).zfill(4) + '.png'
    my1dout  = 'my1d.'          + repr(tstep).zfill(4) + '.png'
    sp1dout  = 'speed1d.'       + repr(tstep).zfill(4) + '.png'

    # set x and z meshgrid objects
    X,Y = np.meshgrid(xgrid,ygrid)

    # surface plots
    numfigs += 1
    fig = plt.figure(numfigs)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, rho[:,:,nz//2,tstep], rstride=1, cstride=1, 
                    cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlim((minmaxrho[0], minmaxrho[1]))
    ax.view_init(elevation,angle);
    plt.title(r'$\rho(x,y)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
    plt.savefig(rhosurf)
            
    numfigs += 1
    fig = plt.figure(numfigs)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, et[:,:,nz//2,tstep], rstride=1, cstride=1, 
                    cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlim((minmaxet[0], minmaxet[1]))
    ax.view_init(elevation,angle);
    plt.title(r'$e_t(x,y)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
    plt.savefig(etsurf)
    
    # stream plots
    numfigs += 1
    fig = plt.figure(numfigs,figsize=(12,4))
    ax1 = fig.add_subplot(121)
    lw = speed / speed.max()
    ax1.streamplot(X, Y, U, V, color='b', linewidth=lw)
    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_aspect('equal')
    ax2 = fig.add_subplot(122)
    lw = speedtrue / speedtrue.max()
    ax2.streamplot(X, Y, Utrue, Vtrue, color='k', linewidth=lw)
    ax2.set_xlabel('z'); ax2.set_ylabel('x'); ax2.set_aspect('equal')
    plt.suptitle(r'$\mathbf{v}(x,y)$ (left) vs $\mathbf{v}_{true}(x,y)$ (right) at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
    plt.savefig(vstr)
            
    # contour plots
    # numfigs += 1
    # fig = plt.figure(numfigs,figsize=(12,4))
    # ax1 = fig.add_subplot(121)
    # ax1.contourf(X, Y, rho[:,:,nz//2,tstep])
    # ax1.colorbar();  ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_axis('equal')
    # ax2 = fig.add_subplot(122)
    # ax2.contourf(X, Y, rhotrue)
    # ax2.colorbar();  ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_axis('equal')
    # plt.suptitle(r'$\rho(x,y)$ (left) vs $\rho_{true}(x,y)$ (right) at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
    # plt.savefig(rhotr)
    
    # numfigs += 1
    # fig = plt.figure(numfigs)
    # plt.contourf(X, Y, et[:,:,nz//2,tstep])
    # plt.colorbar();  plt.xlabel('x'); plt.ylabel('y'); plt.axis('equal')
    # plt.title(r'$e_t(x,y)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
    # plt.savefig(etcont)

    
    # line/error plots
    rho1d = rho[:,ny//2,nz//2,tstep]
    mx1d  = mx[:,ny//2,nz//2,tstep]
    my1d  = mz[:,ny//2,nz//2,tstep]
    sp1d  = speed[:,ny//2]
    rhotrue1d = rhotrue[:,ny//2]
    mxtrue1d  = mxtrue[:,ny//2]
    mytrue1d  = mytrue[:,ny//2]
    sptrue1d  = speedtrue[:,ny//2]

    numfigs += 1
    fig = plt.figure(numfigs,figsize=(12,4))
    ax1 = fig.add_subplot(121)
    ax1.plot(xgrid,rho1d,'b--',xgrid,rhotrue1d,'k-')
    ax1.legend(('computed','analytical'))
    ax1.set_xlabel('x'); ax1.set_ylabel(r'$\rho(x)$')
    ax2 = fig.add_subplot(122)
    ax2.semilogy(xgrid,np.abs(rho1d-rhotrue1d)+1e-16)
    ax1.set_xlabel('x'); ax1.set_ylabel(r'$|\rho-\rho_{true}|$')
    plt.suptitle(r'$\rho(x)$ and error at output ' + tstr + ', mesh = ' + nxstr)
    plt.savefig(rho1dout)
    
    numfigs += 1
    fig = plt.figure(numfigs,figsize=(12,4))
    ax1 = fig.add_subplot(121)
    ax1.plot(xgrid,mx1d,'b--',xgrid,mxtrue1d,'k-')
    ax1.legend(('computed','analytical'))
    ax1.set_xlabel('x'); ax1.set_ylabel(r'$m_x(x)$')
    ax2 = fig.add_subplot(122)
    ax2.semilogy(xgrid,np.abs(mx1d-mxtrue1d)+1e-16)
    ax2.set_xlabel('x'); ax2.set_ylabel(r'$|m_x-m_{x,true}|$')
    plt.suptitle(r'$m_x(x)$ and error at output ' + tstr + ', mesh = ' + nxstr)
    plt.savefig(mx1dout)
    
    numfigs += 1
    fig = plt.figure(numfigs,figsize=(12,4))
    ax1 = fig.add_subplot(121)
    ax1.plot(xgrid,my1d,'b--',xgrid,mytrue1d,'k-')
    ax1.legend(('computed','analytical'))
    ax1.set_xlabel('x'); ax1.set_ylabel(r'$m_y(x)$')
    ax2 = fig.add_subplot(122)
    ax2.semilogy(xgrid,np.abs(my1d-mytrue1d)+1e-16)
    ax2.set_xlabel('x'); ax2.set_ylabel(r'$|m_y-m_{y,true}|$')
    plt.suptitle(r'$m_y(x)$ and error at output ' + tstr + ', mesh = ' + nxstr)
    plt.savefig(my1dout)

    numfigs += 1
    fig = plt.figure(numfigs,figsize=(12,4))
    ax1 = fig.add_subplot(121)
    ax1.plot(xgrid,sp1d,'b--',xgrid,sptrue1d,'k-')
    ax1.legend(('computed','analytical'))
    ax1.set_xlabel('x'); ax1.set_ylabel('s(x)')
    ax2 = fig.add_subplot(122)
    ax2.semilogy(xgrid,np.abs(sp1d-sptrue1d)+1e-16)
    ax2.set_xlabel('x'); ax2.set_ylabel(r'$|s-s_{true}|$')
    plt.suptitle(r'$s(x)$ and error at output ' + tstr + ', mesh = ' + nxstr)
    plt.savefig(sp1dout)
    
    if (showplots):
      plt.show()
    for i in range(1,numfigs+1):
      plt.figure(i), plt.close()
        
  
##### end of script #####
