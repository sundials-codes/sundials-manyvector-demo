#!/usr/bin/env python3
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ SMU
#------------------------------------------------------------
# Copyright (c) 2019, Southern Methodist University.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------
# matplotlib-based plotting utility function for
# Rayleigh-Taylor test problem

# imports
import os
import multiprocessing
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from utilities_euler3D import *


# function to generate plots of solution snapshot
def plot_step(tstep):
  numfigs = tstep*100
    
  print('  time step', tstep+1, 'out of', nt)

  # set string constants for current time, mesh sizes
  tstr  = repr(tstep)
  nxstr = repr(nx)
  nystr = repr(ny)
    
  # extract 2D fields
  rho_step = np.transpose(rho[:,:,nz//2,tstep])
  mx_step  = np.transpose( mx[:,:,nz//2,tstep])
  my_step  = np.transpose( my[:,:,nz//2,tstep])
  et_step  = np.transpose( et[:,:,nz//2,tstep])
  U = mx_step/rho_step
  V = my_step/rho_step
  speed = np.sqrt(U**2 + V**2)
    
  # set filenames for graphics
  rhosurf  = 'rho_surface.'   + repr(tstep).zfill(4) + '.png'
  etsurf   = 'et_surface.'    + repr(tstep).zfill(4) + '.png'
  vstr     = 'velocity.'      + repr(tstep).zfill(4) + '.png'
  rhocont  = 'rho_contour.'   + repr(tstep).zfill(4) + '.png'
  etcont   = 'et_contour.'    + repr(tstep).zfill(4) + '.png'

  # set x and z meshgrid objects
  X,Y = np.meshgrid(xgrid,ygrid)

  # surface plots
  numfigs += 1
  fig = plt.figure(numfigs)
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(X, Y, rho_step, rstride=1, cstride=1, 
                  cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
  ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlim((minmaxrho[0], minmaxrho[1]))
  ax.view_init(elevation,angle);
  plt.title(r'$\rho(x,y)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
  plt.savefig(rhosurf)
            
  numfigs += 1
  fig = plt.figure(numfigs)
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(X, Y, et_step, rstride=1, cstride=1, 
                  cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
  ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlim((minmaxet[0], minmaxet[1]))
  ax.view_init(elevation,angle);
  plt.title(r'$e_t(x,y)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
  plt.savefig(etsurf)
    
  # stream plots
  numfigs += 1
  fig = plt.figure(numfigs,figsize=(4,6))
  lw = speed / speed.max()
  plt.streamplot(X, Y, U, V, color='b', linewidth=lw)
  plt.xlabel('x'); plt.ylabel('y'); plt.axis('scaled')
  plt.suptitle(r'$\mathbf{v}(x,y)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
  plt.savefig(vstr)
            
  # contour plots
  numfigs += 1
  fig = plt.figure(numfigs,figsize=(4,6))
  #plt.contourf(X, Y, rho_step); plt.colorbar()
  #plt.pcolor(X, Y, rho_step, vmin=minmaxrho[0], vmax=minmaxrho[1]); plt.colorbar() 
  plt.pcolormesh(X, Y, rho_step, vmin=minmaxrho[0], vmax=minmaxrho[1]); plt.colorbar() 
  plt.xlabel('x'); plt.ylabel('y'); plt.axis('scaled')
  plt.title(r'$\rho(x,y)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
  plt.savefig(rhocont)
    
  numfigs += 1
  fig = plt.figure(numfigs,figsize=(4,6))
  #plt.contourf(X, Y, et_step); plt.colorbar() 
  #plt.pcolor(X, Y, et_step, vmin=minmaxet[0], vmax=minmaxet[1]); plt.colorbar() 
  plt.pcolormesh(X, Y, et_step, vmin=minmaxet[0], vmax=minmaxet[1]); plt.colorbar() 
  plt.xlabel('x'); plt.ylabel('y'); plt.axis('scaled')
  plt.title(r'$e_t(x,y)$ at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
  plt.savefig(etcont)

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
  minmaxet  = [0.9*et.min(), 1.1*et.max()]
  if (minmaxet[0] == minmaxet[1]):
    minmaxet += [-0.1, 0.1]

  # spawn processes to generate plots for each time step
  timesteps = range(0,nt)
  nprocs = multiprocessing.cpu_count()//2
  p = multiprocessing.Pool(nprocs)
  p.map(plot_step, timesteps)
  p.close()

  # generate movies of plots
  print('  ')
  print('Converting plots into movies...')
  basenames = [ 'rho_surface', 'et_surface', 'velocity', 'rho_contour', 'et_contour' ]
  for name in basenames:
    cmd = '../../bin/make_movie.py ' + name + '* -name ' + name + ' > /dev/null 2>&1'
    os.system(cmd)
    
  print('  ')
  print('Finished.')

  
##### end of script #####
