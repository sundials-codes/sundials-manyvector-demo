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
    nx, ny, nz, nchem, nt, xgrid, ygrid, zgrid, tgrid, rho, mx, my, mz, et, chem = load_data()

    # output general information to screen
    print('Generating plots for data set:')
    print('  nx:    ', nx)
    print('  ny:    ', ny)
    print('  nz:    ', nz)
    print('  nchem: ', nchem)
    print('  nt:    ', nt)
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
    minmaxchem = np.zeros((nchem,2), dtype=float)
    for ichem in range(nchem):
        minmaxchem[ichem]  = [0.9*chem[:,:,:,ichem,:].min(), 1.1*chem[:,:,:,ichem,:].max()]
        if (minmaxchem[ichem,0] == minmaxchem[ichem,1]):
            minmaxchem[ichem]  = [0.9*minmaxchem[ichem,0], 1.1*minmaxchem[ichem,1]]

    # determine character widths for chemistry field names
    if (nchem < 11):
        cwidth = 1
    elif (nchem < 101):
        cwidth = 2
    elif (nchem < 1001):
        cwidth = 3
    elif (nchem < 10001):
        cwidth = 4
            
    # generate 'slice' plots of solution
    for tstep in range(nt):
        numfigs = 0
        
        print('time step', tstep+1, 'out of', nt)
            
        # set string constants for current time, mesh sizes
        tstr  = "%.4f" %(tgrid[tstep])
        nxstr = repr(nx)
        nystr = repr(ny)
        nzstr = repr(nz)

        # plot x-slices
        if (xslice):
                
            rhoname = 'xslice-rho.' + repr(tstep).zfill(4) + '.png'
            mxname  = 'xslice-mx.'  + repr(tstep).zfill(4) + '.png'
            myname  = 'xslice-my.'  + repr(tstep).zfill(4) + '.png'
            mzname  = 'xslice-mz.'  + repr(tstep).zfill(4) + '.png'
            etname  = 'xslice-et.'  + repr(tstep).zfill(4) + '.png'
            chemname = [];
            for ichem in range(nchem):
                chemname.append('xslice-c' + repr(ichem).zfill(cwidth) +'.'  + repr(tstep).zfill(4) + '.png')

            # set y and z meshgrid objects
            Z,Y = np.meshgrid(zgrid,ygrid)
                
            # plot slices of current solution as surfaces, and save to disk
            numfigs += 1
            fig = plt.figure(numfigs)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Z, Y, rho[nx//2,:,:,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('z'); ax.set_ylabel('y'); ax.set_zlim((minmaxrho[0], minmaxrho[1]))
            ax.view_init(elevation,angle);
            plt.title(r'$\rho(y,z)$ slice at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
            plt.savefig(rhoname)
            
            numfigs += 1
            fig = plt.figure(numfigs)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Z, Y, mx[nx//2,:,:,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('z'); ax.set_ylabel('y'); ax.set_zlim((minmaxmx[0], minmaxmx[1]))
            ax.view_init(elevation,angle);
            plt.title(r'$m_x(y,z)$ slice at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
            plt.savefig(mxname)
            
            numfigs += 1
            fig = plt.figure(numfigs)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Z, Y, my[nx//2,:,:,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('z'); ax.set_ylabel('y'); ax.set_zlim((minmaxmy[0], minmaxmy[1]))
            ax.view_init(elevation,angle);
            plt.title(r'$m_y(y,z)$ slice at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
            plt.savefig(myname)
            
            numfigs += 1
            fig = plt.figure(numfigs)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Z, Y, mz[nx//2,:,:,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('z'); ax.set_ylabel('y'); ax.set_zlim((minmaxmz[0], minmaxmz[1]))
            ax.view_init(elevation,angle);
            plt.title(r'$m_z(y,z)$ slice at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
            plt.savefig(mzname)
            
            numfigs += 1
            fig = plt.figure(numfigs)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Z, Y, et[nx//2,:,:,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('z'); ax.set_ylabel('y'); ax.set_zlim((minmaxet[0], minmaxet[1]))
            ax.view_init(elevation,angle);
            plt.title(r'$e_t(y,z)$ slice at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
            plt.savefig(etname)

            for ichem in range(nchem):
                numfigs += 1
                fig = plt.figure(numfigs)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(Z, Y, chem[nx//2,:,:,ichem,tstep], rstride=1, cstride=1, 
                                cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
                ax.set_xlabel('z'); ax.set_ylabel('y')
                ax.set_zlim((minmaxchem[ichem,0], minmaxchem[ichem,1]))
                ax.view_init(elevation,angle);
                plt.title(r'c' + repr(ichem) + '(y,z) slice at output ' + tstr + ', mesh = ' + nystr + 'x' + nzstr)
                plt.savefig(chemname[ichem])
            
        # plot y-slices
        if (yslice):
                
            rhoname = 'yslice-rho.' + repr(tstep).zfill(4) + '.png'
            mxname  = 'yslice-mx.'  + repr(tstep).zfill(4) + '.png'
            myname  = 'yslice-my.'  + repr(tstep).zfill(4) + '.png'
            mzname  = 'yslice-mz.'  + repr(tstep).zfill(4) + '.png'
            etname  = 'yslice-et.'  + repr(tstep).zfill(4) + '.png'
            chemname = [];
            for ichem in range(nchem):
                chemname.append('yslice-c' + repr(ichem).zfill(cwidth) +'.'  + repr(tstep).zfill(4) + '.png')
            
            # set x and z meshgrid objects
            Z,X = np.meshgrid(zgrid,xgrid)
                
            # plot slices of current solution as surfaces, and save to disk
            numfigs += 1
            fig = plt.figure(numfigs)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Z, X, rho[:,ny//2,:,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('z'); ax.set_ylabel('x'); ax.set_zlim((minmaxrho[0], minmaxrho[1]))
            ax.view_init(elevation,270+angle);
            plt.title(r'$\rho(x,z)$ slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
            plt.savefig(rhoname)
            
            numfigs += 1
            fig = plt.figure(numfigs)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Z, X, mx[:,ny//2,:,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('z'); ax.set_ylabel('x'); ax.set_zlim((minmaxmx[0], minmaxmx[1]))
            ax.view_init(elevation,270+angle);
            plt.title(r'$m_x(x,z)$ slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
            plt.savefig(mxname)
            
            numfigs += 1
            fig = plt.figure(numfigs)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Z, X, my[:,ny//2,:,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('z'); ax.set_ylabel('x'); ax.set_zlim((minmaxmy[0], minmaxmy[1]))
            ax.view_init(elevation,270+angle);
            plt.title(r'$m_y(x,z)$ slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
            plt.savefig(myname)
            
            numfigs += 1
            fig = plt.figure(numfigs)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Z, X, mz[:,ny//2,:,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('z'); ax.set_ylabel('x'); ax.set_zlim((minmaxmz[0], minmaxmz[1]))
            ax.view_init(elevation,270+angle);
            plt.title(r'$m_z(x,z)$ slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
            plt.savefig(mzname)
            
            numfigs += 1
            fig = plt.figure(numfigs)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Z, X, et[:,ny//2,:,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('z'); ax.set_ylabel('x'); ax.set_zlim((minmaxet[0], minmaxet[1]))
            ax.view_init(elevation,270+angle);
            plt.title(r'$e_t(x,z)$ slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
            plt.savefig(etname)

            for ichem in range(nchem):
                numfigs += 1
                fig = plt.figure(numfigs)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(Z, X, chem[:,ny//2,:,ichem,tstep], rstride=1, cstride=1, 
                                cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
                ax.set_xlabel('z'); ax.set_ylabel('x')
                ax.set_zlim((minmaxchem[ichem,0], minmaxchem[ichem,1]))
                ax.view_init(elevation,angle);
                plt.title(r'c' + repr(ichem) + '(x,z) slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nzstr)
                plt.savefig(chemname[ichem])
            
            
        # plot z-slices
        if (zslice):
            rhoname = 'zslice-rho.' + repr(tstep).zfill(4) + '.png'
            mxname  = 'zslice-mx.'  + repr(tstep).zfill(4) + '.png'
            myname  = 'zslice-my.'  + repr(tstep).zfill(4) + '.png'
            mzname  = 'zslice-mz.'  + repr(tstep).zfill(4) + '.png'
            etname  = 'zslice-et.'  + repr(tstep).zfill(4) + '.png'
            chemname = [];
            for ichem in range(nchem):
                chemname.append('zslice-c' + repr(ichem).zfill(cwidth) +'.'  + repr(tstep).zfill(4) + '.png')
                
            # set x and z meshgrid objects
            Y,X = np.meshgrid(ygrid,xgrid)

            # plot slices of current solution as surfaces, and save to disk
            numfigs += 1
            fig = plt.figure(numfigs)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Y, X, rho[:,:,nz//2,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('y'); ax.set_ylabel('x'); ax.set_zlim((minmaxrho[0], minmaxrho[1]))
            ax.view_init(elevation,angle);
            plt.title(r'$\rho(x,y)$ slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
            plt.savefig(rhoname)
            
            numfigs += 1
            fig = plt.figure(numfigs)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Y, X, mx[:,:,nz//2,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('y'); ax.set_ylabel('x'); ax.set_zlim((minmaxmx[0], minmaxmx[1]))
            ax.view_init(elevation,angle);
            plt.title(r'$m_x(x,y)$ slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
            plt.savefig(mxname)
            
            numfigs += 1
            fig = plt.figure(numfigs)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Y, X, my[:,:,nz//2,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('y'); ax.set_ylabel('x'); ax.set_zlim((minmaxmy[0], minmaxmy[1]))
            ax.view_init(elevation,angle);
            plt.title(r'$m_y(x,y)$ slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
            plt.savefig(myname)
            
            numfigs += 1
            fig = plt.figure(numfigs)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Y, X, mz[:,:,nz//2,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('y'); ax.set_ylabel('x'); ax.set_zlim((minmaxmz[0], minmaxmz[1]))
            ax.view_init(elevation,angle);
            plt.title(r'$m_z(x,y)$ slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
            plt.savefig(mzname)
            
            numfigs += 1
            fig = plt.figure(numfigs)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Y, X, et[:,:,nz//2,tstep], rstride=1, cstride=1, 
                            cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
            ax.set_xlabel('y'); ax.set_ylabel('x'); ax.set_zlim((minmaxet[0], minmaxet[1]))
            ax.view_init(elevation,angle);
            plt.title(r'$e_t(x,y)$ slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
            plt.savefig(etname)

            for ichem in range(nchem):
                numfigs += 1
                fig = plt.figure(numfigs)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(Y, X, chem[:,:,nz//2,ichem,tstep], rstride=1, cstride=1, 
                                cmap=cm.jet, linewidth=0, antialiased=True, shade=True)
                ax.set_xlabel('y'); ax.set_ylabel('x')
                ax.set_zlim((minmaxchem[ichem,0], minmaxchem[ichem,1]))
                ax.view_init(elevation,angle);
                plt.title(r'c' + repr(ichem) + '(x,y) slice at output ' + tstr + ', mesh = ' + nxstr + 'x' + nystr)
                plt.savefig(chemname[ichem])
            
        if (showplots):
            plt.show()
        for i in range(1,numfigs+1):
            plt.figure(i), plt.close()



# run "plot_slices" with default arguments if run from the command line
if __name__== "__main__":
  plot_slices()
  
##### end of script #####
