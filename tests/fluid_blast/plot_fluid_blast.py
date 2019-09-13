#!/usr/bin/env python3
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ SMU
#------------------------------------------------------------
# Copyright (c) 2019, Southern Methodist University.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------
# matplotlib-based plotting utility function for
# fluid blast test problem

# imports
import os
import multiprocessing
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from utilities_euler3D import *


# function to compute the temperature from the total energy
def compute_temperature(rho, mx, my, mz, et):
  gamma = 5.0/3.0
  kboltz = 1.3806488e-16
  mH = 1.67e-24
  N = 1.0/mH*(0.76/1.00794 + 0.24/4.002602)
  return ((gamma-1)/(kboltz*N)*(et - (mx*mx+my*my+mz*mz)*0.5/rho))


# function to generate plots of solution snapshot
def plot_step(tstep):
  numfigs = tstep*100
    
  print('  time step', tstep+1, 'out of', nt)

  # set string constants for current time, mesh sizes
  tstr  = repr(tstep)
  nxstr = repr(nx)
  nystr = repr(ny)
  nzstr = repr(nz)
    
  # extract 2D field projections
  rho_step = np.transpose(rho[:,:,nz//2,tstep])
  mx_step  = np.transpose( mx[:,:,nz//2,tstep])
  my_step  = np.transpose( my[:,:,nz//2,tstep])
  mz_step  = np.transpose( mz[:,:,nz//2,tstep])
  et_step  = np.transpose( et[:,:,nz//2,tstep])
  # rho_step = np.transpose(np.mean(rho[:,:,:,tstep], axis=2))
  # mx_step  = np.transpose(np.mean( mx[:,:,:,tstep], axis=2))
  # my_step  = np.transpose(np.mean( my[:,:,:,tstep], axis=2))
  # mz_step  = np.transpose(np.mean( mz[:,:,:,tstep], axis=2))
  # et_step  = np.transpose(np.mean( et[:,:,:,tstep], axis=2))
  U = mx_step/rho_step
  V = my_step/rho_step
  W = mz_step/rho_step
  T = compute_temperature(rho_step, mx_step, my_step, mz_step, et_step)
  speed = np.sqrt(U**2 + V**2 + W**2)
    
  # set filenames for graphics
  rhosurf  = 'rho_surface.'   + repr(tstep).zfill(4) + '.png'
  etsurf   = 'et_surface.'    + repr(tstep).zfill(4) + '.png'
  Tsurf    = 'T_surface.'     + repr(tstep).zfill(4) + '.png'
  vstr     = 'velocity.'      + repr(tstep).zfill(4) + '.png'
  rhocont  = 'rho_contour.'   + repr(tstep).zfill(4) + '.png'
  etcont   = 'et_contour.'    + repr(tstep).zfill(4) + '.png'
  Tcont    = 'T_contour.'     + repr(tstep).zfill(4) + '.png'

  # set x and z meshgrid objects
  X,Y = np.meshgrid(xgrid,ygrid)
  zstr = "%.2f" % zgrid[nz//2]
  
  # surface plots
  numfigs += 1
  fig = plt.figure(numfigs)
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(X, Y, rho_step, rstride=1, cstride=1, 
                  cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
  ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlim((minmaxrho[0], minmaxrho[1]))
  ax.view_init(elevation,angle);
  # plt.title(r'$\rho(x,y,:)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr + 'x' + nzstr)
  plt.title(r'$\rho(x,y,' + zstr + ')$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr + 'x' + nzstr)
  plt.savefig(rhosurf)
            
  numfigs += 1
  fig = plt.figure(numfigs)
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(X, Y, et_step, rstride=1, cstride=1, 
                  cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
  ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlim((minmaxet[0], minmaxet[1]))
  ax.view_init(elevation,angle);
  # plt.title(r'$e_t(x,y,:)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr + 'x' + nzstr)
  plt.title(r'$e_t(x,y,' + zstr + ')$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr + 'x' + nzstr)
  plt.savefig(etsurf)
    
  numfigs += 1
  fig = plt.figure(numfigs)
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(X, Y, T, rstride=1, cstride=1, 
                  cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
  ax.set_xlabel('x'); ax.set_ylabel('y')
  ax.view_init(elevation,angle);
  # plt.title(r'$T(x,y,:)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr + 'x' + nzstr)
  plt.title(r'$T(x,y,' + zstr + ')$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr + 'x' + nzstr)
  plt.savefig(Tsurf)
    
  # stream plots
  numfigs += 1
  fig = plt.figure(numfigs)
  lw = speed / speed.max()
  plt.streamplot(X, Y, U, V, color='b', linewidth=lw)
  plt.xlabel('x'); plt.ylabel('y'); plt.axis('scaled')
  # plt.suptitle(r'$\mathbf{v}(x,y,:)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr + 'x' + nzstr)
  plt.suptitle(r'$\mathbf{v}(x,y,' + zstr + ')$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr + 'x' + nzstr)
  plt.savefig(vstr)
            
  # contour plots
  numfigs += 1
  fig = plt.figure(numfigs)
  plt.pcolormesh(X, Y, rho_step, vmin=minmaxrho[0], vmax=minmaxrho[1]); plt.colorbar() 
  plt.xlabel('x'); plt.ylabel('y'); plt.axis('scaled')
  # plt.title(r'$\rho(x,y,:)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr + 'x' + nzstr)
  plt.title(r'$\rho(x,y,' + zstr + ')$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr + 'x' + nzstr)
  plt.savefig(rhocont)
    
  numfigs += 1
  fig = plt.figure(numfigs)
  plt.pcolormesh(X, Y, et_step, vmin=minmaxet[0], vmax=minmaxet[1]); plt.colorbar() 
  plt.xlabel('x'); plt.ylabel('y'); plt.axis('scaled')
  # plt.title(r'$e_t(x,y,:)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr + 'x' + nzstr)
  plt.title(r'$e_t(x,y,' + zstr + ')$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr + 'x' + nzstr)
  plt.savefig(etcont)
    
  numfigs += 1
  fig = plt.figure(numfigs)
  plt.pcolormesh(X, Y, T); plt.colorbar() 
  plt.xlabel('x'); plt.ylabel('y'); plt.axis('scaled')
  # plt.title(r'$T(x,y,:)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr + 'x' + nzstr)
  plt.title(r'$T(x,y,' + zstr + ')$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr + 'x' + nzstr)
  plt.savefig(Tcont)

  for i in range(1,numfigs+1):
    plt.figure(i), plt.close()
  return


if __name__ == '__main__':

  # set view for surface plots  
  elevation = 15
  angle = 20

  # load solution data
  print('  ')
  print('Loading data...')
  nx, ny, nz, nchem, nt, xgrid, ygrid, zgrid, tgrid, rho, mx, my, mz, et, chem = load_data()

  # output general information to screen
  print('  ')
  print('Generating plots for data set...')
  print('  nx: ', nx)
  print('  ny: ', ny)
  print('  nz: ', nz)
  print('  nt: ', nt)
  
  # determine extents of plots
  minmaxrho = [0.9*rho.min(), 1.1*rho.max()]
  if (minmaxrho[0] == minmaxrho[1]):
    minmaxrho += [-0.1, 0.1]
  minmaxmx  = [0.9*mx.min(), 1.1*mx.max()]
  if (minmaxmx[0] == minmaxmx[1]):
    minmaxmx += [-0.1, 0.1]
  minmaxmy  = [0.9*my.min(), 1.1*my.max()]
  if (minmaxmy[0] == minmaxmy[1]):
    minmaxmy += [-0.1, 0.1]
  minmaxmz  = [0.9*mz.min(), 1.1*mz.max()]
  if (minmaxmz[0] == minmaxmz[1]):
    minmaxmz += [-0.1, 0.1]
  minmaxet  = [0.9*et.min(), 1.1*et.max()]
  if (minmaxet[0] == minmaxet[1]):
    minmaxet += [-0.1, 0.1]

  # output min/max values for each field
  print('  ')
  print('Minimum/maximum values for each field:')
  print('   rho: ', minmaxrho)
  print('    mx: ', minmaxmx)
  print('    my: ', minmaxmy)
  print('    mz: ', minmaxmz)
  print('    et: ', minmaxet)
    
  # spawn processes to generate plots for each time step
  timesteps = range(0,nt)
  nprocs = max(1, multiprocessing.cpu_count()//8)
  p = multiprocessing.Pool(nprocs)
  p.map(plot_step, timesteps)
  p.close()

  # generate movies of plots
  print('  ')
  print('Converting plots into movies...')
  basenames = [ 'rho_surface', 'et_surface', 'T_surface', 'velocity', 'rho_contour', 'et_contour', 'T_contour' ]
  for name in basenames:
    cmd = '../../bin/make_movie.py ' + name + '* -name ' + name + ' > /dev/null 2>&1'
    os.system(cmd)
    
  print('  ')
  print('Finished.')

  
##### end of script #####
