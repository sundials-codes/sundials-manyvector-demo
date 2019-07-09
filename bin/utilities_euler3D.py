#!/usr/bin/env python3
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ SMU
#------------------------------------------------------------
# Copyright (c) 2019, Southern Methodist University.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------
# numpy-based utility routines for euler3D.cpp code

# imports
import sys
import numpy as np

# load solution data into multi-dimensional arrays
def load_data():
    import h5py

    # load from restart_parameters.txt
    f = open('restart_parameters.txt')
    d = {}
    for line in f:
        linevals = line.split()
        if linevals: d.update({linevals[0]: linevals[2]})
    nx = int(d['nx'])
    ny = int(d['ny'])
    nz = int(d['nz'])
    nt = int(d['restart'])+1

    # create spatial mesh arrays
    xgrid = np.linspace(float(d['xl']), float(d['xr']), nx)
    ygrid = np.linspace(float(d['yl']), float(d['yr']), ny)
    zgrid = np.linspace(float(d['zl']), float(d['zr']), nz)
    tgrid = np.zeros((nt+1), dtype=float)

    # read first output file, and verify that it matches metadata
    f = h5py.File("output-0000000.hdf5", "r")
    if ( (nx != np.size(f['Density'][:], 2)) or
         (ny != np.size(f['Density'][:], 1)) or
         (nz != np.size(f['Density'][:], 0)) ):
        raise ValueError("load_data: HDF5 file output-0000000.hdf5 and metadata disagree on data shape")

    # set number of chemical species and current time
    nchem = f[('nchem')].value
    tgrid[0] = f[('time')].value

    # create empty arrays for all solution data
    rho  = np.zeros((nx,ny,nz,nt), order='F')
    mx   = np.zeros((nx,ny,nz,nt), order='F')
    my   = np.zeros((nx,ny,nz,nt), order='F')
    mz   = np.zeros((nx,ny,nz,nt), order='F')
    et   = np.zeros((nx,ny,nz,nt), order='F')
    chem = np.zeros((nx,ny,nz,nchem,nt), order='F')

    # insert first dataset results arrays
    rho[:,:,:,0] = np.transpose(f['Density'])
    mx[ :,:,:,0] = np.transpose(f['x-Momentum'])
    my[ :,:,:,0] = np.transpose(f['y-Momentum'])
    mz[ :,:,:,0] = np.transpose(f['z-Momentum'])
    et[ :,:,:,0] = np.transpose(f['TotalEnergy'])
    for ichem in range(nchem):
        chemname = 'Chemical-' + repr(ichem).zfill(3)
        chem[:,:,:,ichem,0] = np.transpose(f[chemname])

    # iterate over remaining hdf5 files, inserting into output
    for iout in range(1,nt):
        fname = 'output-' + repr(iout).zfill(7) + '.hdf5'
        f = h5py.File(fname, 'r')
        tgrid[iout] = f[('time')].value

        # check that this file's dimensions match the first one
        if ( (nx != np.size(f['Density'][:], 2)) or
             (ny != np.size(f['Density'][:], 1)) or
             (nz != np.size(f['Density'][:], 0)) ):
            raise ValueError("load_data: HDF5 file " + fname + " and metadata disagree on data shape")
        if (nchem != f[('nchem')].value):
            raise ValueError("load_data: HDF5 file " + fname + " and metadata disagree on nchem")

        # insert into output arrays
        rho[:,:,:,iout] = np.transpose(f['Density'])
        mx[ :,:,:,iout] = np.transpose(f['x-Momentum'])
        my[ :,:,:,iout] = np.transpose(f['y-Momentum'])
        mz[ :,:,:,iout] = np.transpose(f['z-Momentum'])
        et[ :,:,:,iout] = np.transpose(f['TotalEnergy'])
        for ichem in range(nchem):
            chemname = 'Chemical-' + repr(ichem).zfill(3)
            chem[:,:,:,ichem,iout] = np.transpose(f[chemname])

    return [nx, ny, nz, nchem, nt, xgrid, ygrid, zgrid, tgrid, rho, mx, my, mz, et, chem]



def fsecant(p4, p1, p5, rho1, rho5, gamma):
    """
    f = fsecant(p4, p1, p5, rho1, rho5, gamma)

    Utility routine for exact_Riemann function
    """
    z = (p4 / p5 - 1.0)
    c1 = np.sqrt(gamma * p1 / rho1)
    c5 = np.sqrt(gamma * p5 / rho5)

    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    g2  = 2.0 * gamma

    fact = gm1 / g2 * (c5 / c1) * z / np.sqrt (1. + gp1 / g2 * z)
    fact = (1.0 - fact)**(g2 / gm1)

    f = p1 * fact - p4

    return f



def exact_Riemann(t, x, xI, gamma, rhoL, rhoR, uL, uR, pL, pR):
    """
    rho, u, m, p, et = exact_Riemann(t, x, xI, gamma, rhoL, rhoR, uL, uR, pL, pR)

    Exact 1D Riemann problem solver (retrieves domain from UserData structure),
    based on Fortran code at http://cococubed.asu.edu/codes/riemann/exact_riemann.f

    Inputs: t time for desired solution,
            x spatial grid for solution output
            [xL,xR] spatial domain,
            xI location of discontinuity at t=0,
            gamma parameter for gas equation of state
            rhoL, rhoR -- left/right densities for Riemann problem
            uL, uR -- left/right velocities for Riemann problem
            pL, pR -- left/right pressures for Riemann problem
    Outputs: density (rho), velocity (u) and pressure (p) over grid x at t
    """

    # number of points in solution
    npts = np.size(x)

    # initialze outputs
    rho = np.zeros(npts, dtype=float)
    u   = np.zeros(npts, dtype=float)
    m   = np.zeros(npts, dtype=float)
    p   = np.zeros(npts, dtype=float)
    et  = np.zeros(npts, dtype=float)

    # begin solution
    if (pL > pR):
        rho1 = rhoL
        p1   = pL
        u1   = uL
        rho5 = rhoR
        p5   = pR
        u5   = uR
    else:
        rho1 = rhoR
        p1   = pR
        u1   = uR
        rho5 = rhoL
        p5   = pL
        u5   = uL

    # solve for post-shock pRessure by secant method
    p40 = p1
    p41 = p5
    f0 = fsecant(p40, p1, p5, rho1, rho5, gamma)
    itmax = 50
    eps   = 1.e-14

    for iter in range(1,itmax+1):
        f1 = fsecant(p41, p1, p5, rho1, rho5, gamma)
        if (f1 == f0):
            break
        p4 = p41 - (p41 - p40) * f1 / (f1 - f0)
        error = np.abs(p4 - p41) / np.abs(p41)
        if (error < eps):
            break
        p40 = p41
        p41 = p4
        f0  = f1
        if (iter == itmax):
            write('exact_Riemann iteration failed to converge')
            return [rho, u, m, p, et]

    # compute post-shock density and velocity
    z  = (p4 / p5 - 1.0)
    c5 = np.sqrt(gamma * p5 / rho5)

    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    gmfac1 = 0.5 * gm1 / gamma
    gmfac2 = 0.5 * gp1 / gamma

    fact = np.sqrt(1.0 + gmfac2 * z)

    u4 = c5 * z / (gamma * fact)
    rho4 = rho5 * (1.0 + gmfac2 * z) / (1.0 + gmfac1 * z)

    # shock speed
    w = c5 * fact

    # compute values at foot of rarefaction
    p3 = p4
    u3 = u4
    rho3 = rho1 * (p3 / p1)**(1.0 /gamma)

    # compute positions of waves
    if (pL > pR):
        c1 = np.sqrt(gamma * p1 / rho1)
        c3 = np.sqrt(gamma * p3 / rho3)

        xsh = xI + w * t
        xcd = xI + u3 * t
        xft = xI + (u3 - c3) * t
        xhd = xI - c1 * t

        # compute solution as a function of position
        for i in range(npts):
            if (x[i] < xhd):
                rho[i] = rho1
                p[i]   = p1
                u[i]   = u1
            elif (x[i] < xft):
                u[i]   = 2.0 / gp1 * (c1 + (x[i] - xI) / t)
                fact   = 1.0 - 0.5 * gm1 * u[i] / c1
                rho[i] = rho1 * fact**(2.0 / gm1)
                p[i]   = p1 * fact**(2.0 * gamma / gm1)
            elif (x[i] < xcd):
                rho[i] = rho3
                p[i]   = p3
                u[i]   = u3
            elif (x[i] < xsh):
                rho[i] = rho4
                p[i]   = p4
                u[i]   = u4
            else:
                rho[i] = rho5
                p[i]   = p5
                u[i]   = u5

    # if pR > pL, reverse solution
    if (pR > pL):
        c1 = np.sqrt(gamma * p1 / rho1)
        c3 = np.sqrt(gamma * p3 / rho3)

        xsh = xI - w * t
        xcd = xI - u3 * t
        xft = xI - (u3 - c3) * t
        xhd = xI + c1 * t

        # compute solution as a function of position
        for i in range(npts):
            if (x[i] < xsh):
                rho[i] = rho5
                p[i]   = p5
                u[i]   = -u5
            elif (x[i] < xcd):
                rho[i] = rho4
                p[i]   = p4
                u[i]   = -u4
            elif (x[i] < xft):
                rho[i] = rho3
                p[i]   = p3
                u[i]   = -u3
            elif (x[i] < xhd):
                u[i]   = -2.0 / gp1 * (c1 + (xI - x[i]) / t)
                fact   = 1.0 + 0.5 * gm1 * u[i] / c1
                rho[i] = rho1 * fact**(2.0 / gm1)
                p[i]   = p1 * fact**(2.0 * gamma / gm1)
            else:
                rho[i] = rho1
                p[i]   = p1
                u[i]   = -u1

    # return results
    m = rho*u
    et = p/gm1 + 0.5*u*u*rho
    return [rho, u, m, p, et]

##### end of script #####
