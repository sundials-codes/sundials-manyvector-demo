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
nx, ny, nz, nchem, nt, xgrid, ygrid, zgrid, tgrid, rho, mx, my, mz, et, chem = load_data()

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
    numfigs = 0
    
    print('time step', tstep+1, 'out of', nt)

    # get true solutions
    rhotrue, mztrue, mxtrue = analytical_solution(tgrid[tstep],nz,nx)
    rhotrue = np.transpose(rhotrue)
    mztrue = np.transpose(mztrue)
    mxtrue = np.transpose(mxtrue)
    
    # set string constants for current time, mesh sizes
    tstr  = repr(tstep)
    nzstr = repr(nz)
    nxstr = repr(nx)

    # extract 2D velocity fields (computed and true)
    U = mz[:,ny//2,:,tstep]/rho[:,ny//2,:,tstep]
    Utrue = mztrue/rhotrue
    V = mx[:,ny//2,:,tstep]/rho[:,ny//2,:,tstep]
    Vtrue = mxtrue/rhotrue
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
    mz1dout  = 'mz1d.'          + repr(tstep).zfill(4) + '.png'
    sp1dout  = 'speed1d.'       + repr(tstep).zfill(4) + '.png'

    # set z and x meshgrid objects
    X,Z = np.meshgrid(xgrid,zgrid)

    # surface plots
    numfigs += 1
    fig = plt.figure(numfigs)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Z, rho[:,ny//2,:,tstep], rstride=1, cstride=1, 
                    cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel('x'); ax.set_ylabel('z'); ax.set_zlim((minmaxrho[0], minmaxrho[1]))
    ax.view_init(elevation,angle);
    plt.title(r'$\rho(x,z)$ at output ' + tstr + ', mesh = ' + nxstr + 'z' + nzstr)
    plt.savefig(rhosurf)
            
    numfigs += 1
    fig = plt.figure(numfigs)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Z, et[:,ny//2,:,tstep], rstride=1, cstride=1, 
                    cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel('x'); ax.set_ylabel('z'); ax.set_zlim((minmaxet[0], minmaxet[1]))
    ax.view_init(elevation,angle);
    plt.title(r'$e_t(x,z)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
    plt.savefig(etsurf)
    
    # stream plots
    numfigs += 1
    fig = plt.figure(numfigs,figsize=(12,4))
    ax1 = fig.add_subplot(121)
    lw = speed / speed.max()
    ax1.streamplot(X, Z, U, V, color='b', linewidth=lw)
    ax1.set_xlabel('z'); ax1.set_ylabel('x'); ax1.set_aspect('equal')
    ax2 = fig.add_subplot(122)
    lw = speedtrue / speedtrue.max()
    ax2.streamplot(X, Z, Utrue, Vtrue, color='k', linewidth=lw)
    ax2.set_xlabel('z'); ax2.set_ylabel('x'); ax2.set_aspect('equal')
    plt.suptitle(r'$\mathbf{v}(x,z)$ (left) vs $\mathbf{v}_{true}(x,z)$ (right) at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
    plt.savefig(vstr)
            
    # contour plots
    # numfigs += 1
    # fig = plt.figure(numfigs,figsize=(12,4))
    # ax1 = fig.add_subplot(121)
    # ax1.contourf(X, Z, rho[:,ny//2,:,tstep])
    # ax1.colorbar();  ax1.set_xlabel('x'); ax1.set_ylabel('z'); ax1.set_axis('equal')
    # ax2 = fig.add_subplot(122)
    # ax2.contourf(X, Z, rhotrue)
    # ax2.colorbar();  ax2.set_xlabel('x'); ax2.set_ylabel('z'); ax2.set_axis('equal')
    # plt.suptitle(r'$\rho(x,z)$ (left) vs $\rho_{true}(x,z)$ (right) at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
    # plt.savefig(rhotr)
    
    # numfigs += 1
    # fig = plt.figure(numfigs)
    # plt.contourf(X, Z, et[:,ny//2,:,tstep])
    # plt.colorbar();  plt.xlabel('x'); plt.ylabel('z'); plt.axis('equal')
    # plt.title(r'$e_t(x,z)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
    # plt.savefig(etcont)
    

    # line/error plots
    rho1d = rho[:,ny//2,nz//2,tstep]
    mx1d  = mx[:,ny//2,nz//2,tstep]
    mz1d  = mz[:,ny//2,nz//2,tstep]
    sp1d  = speed[:,nz//2]
    rhotrue1d = rhotrue[:,nz//2]
    mxtrue1d  = mxtrue[:,nz//2]
    mztrue1d  = mztrue[:,nz//2]
    sptrue1d  = speedtrue[:,nz//2]

    numfigs += 1
    fig = plt.figure(numfigs,figsize=(12,4))
    ax1 = fig.add_subplot(121)
    ax1.plot(xgrid,rho1d,'b--',xgrid,rhotrue1d,'k-')
    ax1.legend(('computed','analytical'))
    ax1.set_xlabel('x'); ax1.set_ylabel(r'$\rho(x)$')
    ax2 = fig.add_subplot(122)
    ax2.semilogy(xgrid,np.abs(rho1d-rhotrue1d)+1e-16)
    ax2.set_xlabel('x'); ax2.set_ylabel(r'$|\rho-\rho_{true}|$')
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
    ax1.plot(xgrid,mz1d,'b--',xgrid,mztrue1d,'k-')
    ax1.legend(('computed','analytical'))
    ax1.set_xlabel('x'); ax1.set_ylabel(r'$m_z(x)$')
    ax2 = fig.add_subplot(122)
    ax2.semilogy(xgrid,np.abs(mz1d-mztrue1d)+1e-16)
    ax2.set_xlabel('x'); ax2.set_ylabel(r'$|m_z-m_{z,true}|$')
    plt.suptitle(r'$m_z(x)$ and error at output ' + tstr + ', mesh = ' + nxstr)
    plt.savefig(mz1dout)

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
