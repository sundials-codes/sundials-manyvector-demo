#!/usr/bin/env python3
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ SMU
#------------------------------------------------------------
# Copyright (c) 2019, Southern Methodist University.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------
# matplotlib-based plotting utility function for euler3D.cpp code

# imports
import numpy as np
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from utilities_euler3D import *


def plot_slices(elevation=15, angle=20, slices='all', showplots=False):
    """
    Usage: plot_slices(elevation, angle, slices, showplots)
    
    All inputs are optional:
      elevation, angle -- should be valid arguments for matplotlib's view_init routine
      slices ('all', 'x', 'y', or 'z') -- directional slices to plot, invalid input defaults to 'all'
      showplots (True/False) -- pause and display the plots for each time step before proceeding
    """

    # determine slice plots based on "slices" input
    xslice = True
    yslice = True
    zslice = True
    if (slices == 'x'):
        yslice = False
        zslice = False
    if (slices == 'y'):
        xslice = False
        zslice = False
    if (slices == 'z'):
        xslice = False
        yslice = False
    
    # load solution data
    nx, ny, nz, nt, rho, mx, my, mz, et = load_data()

    # output general information to screen
    print('Generating plots for data set:')
    print('  nx: ', nx)
    print('  ny: ', ny)
    print('  nz: ', nz)
    print('  nt: ', nt)
    if (xslice):
        print('  x-directional slice plots enabled')
    if (yslice):
        print('  y-directional slice plots enabled')
    if (zslice):
        print('  z-directional slice plots enabled')
    
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
    minmaxmz  = [0.9*mz.min(), 1.1*mz.max()]
    if (mz.min() == mz.max()):
        minmaxmz = [mz.min()-0.1, mz.max()+0.1]
    minmaxet  = [0.9*et.min(), 1.1*et.max()]
    if (et.min() == et.max()):
        minmaxet = [et.min()-0.1, et.max()+0.1]

    # generate 'slice' plots of solution
    for tstep in range(nt):

        print('time step', tstep+1, 'out of', nt)
            
        # set string constants for current time, mesh sizes
        tstr  = repr(tstep)
        nxstr = repr(nx)
        nystr = repr(ny)
        nzstr = repr(nz)

        # plot x-slices
        if (xslice):
                
            rhoname = 'xslice-euler3D_rho.' + repr(tstep).zfill(4) + '.png'
            mxname  = 'xslice-euler3D_mx.'  + repr(tstep).zfill(4) + '.png'
            myname  = 'xslice-euler3D_my.'  + repr(tstep).zfill(4) + '.png'
            mzname  = 'xslice-euler3D_mz.'  + repr(tstep).zfill(4) + '.png'
            etname  = 'xslice-euler3D_et.'  + repr(tstep).zfill(4) + '.png'
                
            # set y and z meshgrid objects
            yspan = np.linspace(0.0, 1.0, ny)
            zspan = np.linspace(0.0, 1.0, nz)
            Z,Y = np.meshgrid(zspan,yspan)
                
            # plot slices of current solution as surfaces, and save to disk
            fig = plt.figure(1)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Z, Y, rho[nx//2,:,:,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('z'); ax.set_ylabel('y'); ax.set_zlim((minmaxrho[0], minmaxrho[1]))
            ax.view_init(elevation,angle);
            title(r'$\rho(y,z)$ slice at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
            savefig(rhoname)
            
            fig = plt.figure(2)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Z, Y, mx[nx//2,:,:,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('z'); ax.set_ylabel('y'); ax.set_zlim((minmaxmx[0], minmaxmx[1]))
            ax.view_init(elevation,angle);
            title(r'$m_x(y,z)$ slice at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
            savefig(mxname)
            
            fig = plt.figure(3)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Z, Y, my[nx//2,:,:,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('z'); ax.set_ylabel('y'); ax.set_zlim((minmaxmy[0], minmaxmy[1]))
            ax.view_init(elevation,angle);
            title(r'$m_y(y,z)$ slice at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
            savefig(myname)
            
            fig = plt.figure(4)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Z, Y, mz[nx//2,:,:,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('z'); ax.set_ylabel('y'); ax.set_zlim((minmaxmz[0], minmaxmz[1]))
            ax.view_init(elevation,angle);
            title(r'$m_z(y,z)$ slice at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
            savefig(mzname)
            
            fig = plt.figure(5)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Z, Y, et[nx//2,:,:,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('z'); ax.set_ylabel('y'); ax.set_zlim((minmaxet[0], minmaxet[1]))
            ax.view_init(elevation,angle);
            title(r'$e_t(y,z)$ slice at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
            savefig(etname)
            
            if (showplots):
                plt.show()
            plt.figure(1), plt.close()
            plt.figure(2), plt.close()
            plt.figure(3), plt.close()
            plt.figure(4), plt.close()
            plt.figure(5), plt.close()
            

        # plot y-slices
        if (yslice):
                
            rhoname = 'yslice-euler3D_rho.' + repr(tstep).zfill(4) + '.png'
            mxname  = 'yslice-euler3D_mx.'  + repr(tstep).zfill(4) + '.png'
            myname  = 'yslice-euler3D_my.'  + repr(tstep).zfill(4) + '.png'
            mzname  = 'yslice-euler3D_mz.'  + repr(tstep).zfill(4) + '.png'
            etname  = 'yslice-euler3D_et.'  + repr(tstep).zfill(4) + '.png'
            
            # set x and z meshgrid objects
            xspan = np.linspace(0.0, 1.0, nx)
            zspan = np.linspace(0.0, 1.0, nz)
            Z,X = np.meshgrid(zspan,xspan)
                
            # plot slices of current solution as surfaces, and save to disk
            fig = plt.figure(1)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Z, X, rho[:,ny//2,:,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('z'); ax.set_ylabel('x'); ax.set_zlim((minmaxrho[0], minmaxrho[1]))
            ax.view_init(elevation,270+angle);
            title(r'$\rho(x,z)$ slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
            savefig(rhoname)
            
            fig = plt.figure(2)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Z, X, mx[:,ny//2,:,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('z'); ax.set_ylabel('x'); ax.set_zlim((minmaxmx[0], minmaxmx[1]))
            ax.view_init(elevation,270+angle);
            title(r'$m_x(x,z)$ slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
            savefig(mxname)
            
            fig = plt.figure(3)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Z, X, my[:,ny//2,:,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('z'); ax.set_ylabel('x'); ax.set_zlim((minmaxmy[0], minmaxmy[1]))
            ax.view_init(elevation,270+angle);
            title(r'$m_y(x,z)$ slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
            savefig(myname)
            
            fig = plt.figure(4)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Z, X, mz[:,ny//2,:,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('z'); ax.set_ylabel('x'); ax.set_zlim((minmaxmz[0], minmaxmz[1]))
            ax.view_init(elevation,270+angle);
            title(r'$m_z(x,z)$ slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
            savefig(mzname)
            
            fig = plt.figure(5)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Z, X, et[:,ny//2,:,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('z'); ax.set_ylabel('x'); ax.set_zlim((minmaxet[0], minmaxet[1]))
            ax.view_init(elevation,270+angle);
            title(r'$e_t(x,z)$ slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
            savefig(etname)
            
            if (showplots):
                plt.show()
            plt.figure(1), plt.close()
            plt.figure(2), plt.close()
            plt.figure(3), plt.close()
            plt.figure(4), plt.close()
            plt.figure(5), plt.close()
            
            
        # plot z-slices
        if (zslice):
            rhoname = 'zslice-euler3D_rho.' + repr(tstep).zfill(4) + '.png'
            mxname  = 'zslice-euler3D_mx.'  + repr(tstep).zfill(4) + '.png'
            myname  = 'zslice-euler3D_my.'  + repr(tstep).zfill(4) + '.png'
            mzname  = 'zslice-euler3D_mz.'  + repr(tstep).zfill(4) + '.png'
            etname  = 'zslice-euler3D_et.'  + repr(tstep).zfill(4) + '.png'
                
            # set x and z meshgrid objects
            xspan = np.linspace(0.0, 1.0, nx)
            yspan = np.linspace(0.0, 1.0, ny)
            Y,X = np.meshgrid(yspan,xspan)

            # plot slices of current solution as surfaces, and save to disk
            fig = plt.figure(1)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Y, X, rho[:,:,nz//2,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('y'); ax.set_ylabel('x'); ax.set_zlim((minmaxrho[0], minmaxrho[1]))
            ax.view_init(elevation,angle);
            title(r'$\rho(x,y)$ slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
            savefig(rhoname)
            
            fig = plt.figure(2)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Y, X, mx[:,:,nz//2,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('y'); ax.set_ylabel('x'); ax.set_zlim((minmaxmx[0], minmaxmx[1]))
            ax.view_init(elevation,angle);
            title(r'$m_x(x,y)$ slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
            savefig(mxname)
            
            fig = plt.figure(3)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Y, X, my[:,:,nz//2,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('y'); ax.set_ylabel('x'); ax.set_zlim((minmaxmy[0], minmaxmy[1]))
            ax.view_init(elevation,angle);
            title(r'$m_y(x,y)$ slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
            savefig(myname)
            
            fig = plt.figure(4)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Y, X, mz[:,:,nz//2,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('y'); ax.set_ylabel('x'); ax.set_zlim((minmaxmz[0], minmaxmz[1]))
            ax.view_init(elevation,angle);
            title(r'$m_z(x,y)$ slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
            savefig(mzname)
            
            fig = plt.figure(5)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Y, X, et[:,:,nz//2,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('y'); ax.set_ylabel('x'); ax.set_zlim((minmaxet[0], minmaxet[1]))
            ax.view_init(elevation,angle);
            title(r'$e_t(x,y)$ slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
            savefig(etname)
            
            if (showplots):
                plt.show()
            plt.figure(1), plt.close()
            plt.figure(2), plt.close()
            plt.figure(3), plt.close()
            plt.figure(4), plt.close()
            plt.figure(5), plt.close()
    


# run "plot_slices" with default arguments if run from the command line
if __name__== "__main__":
  plot_slices()
  
##### end of script #####
