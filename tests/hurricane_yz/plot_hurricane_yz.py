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
    numfigs = 0
   
    print('time step', tstep+1, 'out of', nt)
    
    # get true solutions
    rhotrue, mytrue, mztrue = analytical_solution(tgrid[tstep],ny,nz)
    
    # set string constants for current time, mesh sizes
    tstr  = repr(tstep)
    nystr = repr(ny)
    nzstr = repr(nz)

    # extract 2D velocity fields (computed and true)
    U = my[nx//2,:,:,tstep]/rho[nx//2,:,:,tstep]
    Utrue = mytrue/rhotrue
    V = mz[nx//2,:,:,tstep]/rho[nx//2,:,:,tstep]
    Vtrue = mztrue/rhotrue
    speed = np.sqrt(U**2 + V**2)
    speedtrue = np.sqrt(Utrue**2 + Vtrue**2)
    
    # set filenames for graphics
    rhosurf  = 'rho_surface.'   + repr(tstep).zfill(4) + '.png'
    etsurf   = 'et_surface.'    + repr(tstep).zfill(4) + '.png'
    vstr     = 'velocity.'      + repr(tstep).zfill(4) + '.png'
    vtrue    = 'velocity_true.' + repr(tstep).zfill(4) + '.png'
    rhocont  = 'rho_contour.'   + repr(tstep).zfill(4) + '.png'
    rhotr    = 'rho_true.'      + repr(tstep).zfill(4) + '.png'
    etcont   = 'et_contour.'    + repr(tstep).zfill(4) + '.png'
    rho1dout = 'rho1d.'         + repr(tstep).zfill(4) + '.png'
    rho1derr = 'rho1d_error.'   + repr(tstep).zfill(4) + '.png'
    my1dout  = 'my1d.'          + repr(tstep).zfill(4) + '.png'
    my1derr  = 'my1d_error.'    + repr(tstep).zfill(4) + '.png'
    mz1dout  = 'my1d.'          + repr(tstep).zfill(4) + '.png'
    mz1derr  = 'my1d_error.'    + repr(tstep).zfill(4) + '.png'
    sp1dout  = 'speed1d.'       + repr(tstep).zfill(4) + '.png'
    sp1derr  = 'speed1d_error.' + repr(tstep).zfill(4) + '.png'

    # set y and z meshgrid objects
    Z,Y = np.meshgrid(zgrid,ygrid)

    # surface plots
    numfigs += 1
    fig = plt.figure(numfigs)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Z, Y, rho[nx//2,:,:,tstep], rstride=1, cstride=1, 
                    cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel('z'); ax.set_ylabel('y'); ax.set_zlim((minmaxrho[0], minmaxrho[1]))
    ax.view_init(elevation,angle)
    plt.title(r'$\rho(y,z)$ at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
    plt.savefig(rhosurf)

    numfigs += 1
    fig = plt.figure(numfigs)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Z, Y, et[nx//2,:,:,tstep], rstride=1, cstride=1, 
                    cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel('z'); ax.set_ylabel('y'); ax.set_zlim((minmaxet[0], minmaxet[1]))
    ax.view_init(elevation,angle)
    plt.title(r'$e_t(y,z)$ at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
    plt.savefig(etsurf)
    
    # stream plots
    numfigs += 1
    fig = plt.figure(numfigs)
    ax = fig.add_subplot(111)
    lw = speed / speed.max()
    ax.streamplot(Z, Y, U, V, color='b', linewidth=lw)
    ax.set_xlabel('z'); ax.set_ylabel('y'); ax.set_aspect('equal')
    plt.title(r'$\mathbf{v}(y,z)$ at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
    plt.savefig(vstr)
            
    numfigs += 1
    fig = plt.figure(numfigs)
    ax = fig.add_subplot(111)
    lw = speedtrue / speedtrue.max()
    ax.streamplot(Z, Y, Utrue, Vtrue, color='k', linewidth=lw)
    ax.set_xlabel('z'); ax.set_ylabel('y'); ax.set_aspect('equal')
    plt.title(r'$\mathbf{v}_{true}(y,z)$ at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
    plt.savefig(vtrue)
            
    # contour plots
    # numfigs += 1
    # fig = plt.figure(numfigs)
    # plt.contourf(Z, Y, rho[nx//2,:,:,tstep])
    # plt.colorbar();  plt.xlabel('z'); plt.ylabel('y'); plt.axis('equal')
    # plt.title(r'$\rho(y,z)$ at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
    # plt.savefig(rhocont)
            
    # numfigs += 1
    # fig = plt.figure(numfigs)
    # plt.contourf(Z, Y, et[nx//2,:,:,tstep])
    # plt.colorbar();  plt.xlabel('z'); plt.ylabel('y'); plt.axis('equal')
    # plt.title(r'$e_t(y,z)$ at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
    # plt.savefig(etcont)
    
    # numfigs += 1
    # fig = plt.figure(numfigs)
    # plt.contourf(Z, Y, rhotrue)
    # plt.colorbar();  plt.xlabel('z'); plt.ylabel('y'); plt.axis('equal')
    # plt.title(r'$\rho_{true}(y,z)$ at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
    # plt.savefig(rhotr)

    # line/error plots
    rho1d = rho[nx//2,:,nz//2,tstep]
    my1d  = my[nx//2,:,nz//2,tstep]
    mz1d  = mz[nx//2,:,nz//2,tstep]
    sp1d  = speed[:,nz//2]
    rhotrue1d = rhotrue[:,nz//2]
    mytrue1d  = mytrue[:,nz//2]
    mztrue1d  = mztrue[:,nz//2]
    sptrue1d  = speedtrue[:,nz//2]

    numfigs += 1
    fig = plt.figure(numfigs)
    plt.plot(ygrid,rho1d,'b--',ygrid,rhotrue1d,'k-')
    plt.legend(('computed','analytical'))
    plt.xlabel('y'); plt.ylabel(r'$\rho(y)$')
    plt.title(r'$\rho$ slices at output ' + tstr + ', mesh = ' + nystr)
    plt.savefig(rho1dout)

    numfigs += 1
    fig = plt.figure(numfigs)
    plt.semilogy(ygrid,np.abs(rho1d-rhotrue1d)+1e-16)
    plt.xlabel('y'); plt.ylabel('error')
    plt.title(r'$|\rho-\rho_{true}|$ slice at output ' + tstr + ', mesh = ' + nystr)
    plt.savefig(rho1derr)
    
    numfigs += 1
    fig = plt.figure(numfigs)
    plt.plot(ygrid,my1d,'b--',ygrid,mytrue1d,'k-')
    plt.legend(('computed','analytical'))
    plt.xlabel('y'); plt.ylabel(r'$m_y(y)$')
    plt.title(r'$m_y$ slices at output ' + tstr + ', mesh = ' + nystr)
    plt.savefig(my1dout)

    numfigs += 1
    fig = plt.figure(numfigs)
    plt.semilogy(ygrid,np.abs(my1d-mytrue1d)+1e-16)
    plt.xlabel('y'); plt.ylabel('error')
    plt.title(r'$|m_y-m_{y,true}|$ slice at output ' + tstr + ', mesh = ' + nystr)
    plt.savefig(my1derr)
    
    numfigs += 1
    fig = plt.figure(numfigs)
    plt.plot(ygrid,mz1d,'b--',ygrid,mztrue1d,'k-')
    plt.legend(('computed','analytical'))
    plt.xlabel('y'); plt.ylabel(r'$m_z(y)$')
    plt.title(r'$m_z$ slices at output ' + tstr + ', mesh = ' + nystr)
    plt.savefig(mz1dout)

    numfigs += 1
    fig = plt.figure(numfigs)
    plt.semilogy(ygrid,np.abs(mz1d-mztrue1d)+1e-16)
    plt.xlabel('y'); plt.ylabel('error')
    plt.title(r'$|m_z-m_{z,true}|$ slice at output ' + tstr + ', mesh = ' + nystr)
    plt.savefig(mz1derr)
    
    numfigs += 1
    fig = plt.figure(numfigs)
    plt.plot(ygrid,sp1d,'b--',ygrid,sptrue1d,'k-')
    plt.legend(('computed','analytical'))
    plt.xlabel('y'); plt.ylabel('speed(y)')
    plt.title(r'speed slices at output ' + tstr + ', mesh = ' + nystr)
    plt.savefig(sp1dout)

    numfigs += 1
    fig = plt.figure(numfigs)
    plt.semilogy(ygrid,np.abs(sp1d-sptrue1d)+1e-16)
    plt.xlabel('y'); plt.ylabel('error')
    plt.title(r'$|speed-speed_{true}|$ slice at output ' + tstr + ', mesh = ' + nystr)
    plt.savefig(sp1derr)
    
    if (showplots):
      plt.show()
    for i in range(1,numfigs+1):
      plt.figure(i), plt.close()
    
        
  
##### end of script #####
