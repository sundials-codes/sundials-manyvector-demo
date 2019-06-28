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

# determine the number of MPI processes used
def get_MPI_tasks():
    nprocs=1
    for i in range(10000):
        sname = 'output-subdomain.' + repr(i).zfill(7) + '.txt'
        try:
            f = open(sname,'r')
            f.close()
        except IOError:
            nprocs = i
            break
    return nprocs

# load subdomain information, store in table
def load_subdomain_info(nprocs):
    subdomains = np.zeros((nprocs,6), dtype=np.int)
    domain = np.zeros((4,2), dtype=np.float)
    sname = 'output-subdomain.0000000.txt'
    subd = np.loadtxt(sname)
    nx = np.int(subd[0])
    ny = np.int(subd[1])
    nz = np.int(subd[2])
    subdomains[0,:] = subd[3:9]  # [is,ie,js,je,ks,ke]
    nchem = np.int(subd[9])
    domain[0,:] = subd[10:12]     # [xl,xr]
    domain[1,:] = subd[12:14]    # [yl,yr]
    domain[2,:] = subd[14:16]    # [zl,zr]
    domain[3,:] = subd[16:18]    # [t0,tf]
    dTout = np.float(subd[18])
    for i in range(1,nprocs):
        sname = 'output-subdomain.' + repr(i).zfill(7) + '.txt'
        subd = np.loadtxt(sname)
        if ((subd[0] != nx) or (subd[1] != ny) or (subd[2] != nz)):
            sys.exit("error: subdomain files incompatible (clean up and re-run test)")
        subdomains[i,:] = subd[3:9]
    return [nx, ny, nz, nchem, subdomains, domain, dTout]

# load solution data into multi-dimensional arrays
def load_data():

    # load metadata
    nprocs = get_MPI_tasks()
    nx, ny, nz, nchem, subdomains, domain, dTout = load_subdomain_info(nprocs)
    
    # load first processor's data, and determine total number of time steps
    rho_data = np.loadtxt('output-rho.0000000.txt', dtype=np.double)
    mx_data  = np.loadtxt('output-mx.0000000.txt',  dtype=np.double)
    my_data  = np.loadtxt('output-my.0000000.txt',  dtype=np.double)
    mz_data  = np.loadtxt('output-mz.0000000.txt',  dtype=np.double)
    et_data  = np.loadtxt('output-et.0000000.txt',  dtype=np.double)
    chem_data = []
    if (nchem < 11):
        cwidth = 1
    elif (nchem < 101):
        cwidth = 2
    elif (nchem < 1001):
        cwidth = 3
    elif (nchem < 10001):
        cwidth = 4
    for ichem in range(nchem):
        fname = 'output-c' + repr(ichem).zfill(cwidth) + '.0000000.txt'
        chem_data.append(np.loadtxt(fname,  dtype=np.double))
    nt = np.shape(rho_data)[0]
    if ( (np.shape(mx_data)[0] != nt) or (np.shape(my_data)[0] != nt) or
         (np.shape(mz_data)[0] != nt) or (np.shape(et_data)[0] != nt) ):
        sys.exit('error: an output for subdomain 0 has an incorrect number of time steps')
    for ichem in range(nchem):
        if (np.shape(chem_data[ichem])[0] != nt):
            sys.exit('error: an output for subdomain 0 has an incorrect number of time steps')

    # create space-time mesh arrays
    xgrid = np.linspace(domain[0,0], domain[0,1], nx)
    ygrid = np.linspace(domain[1,0], domain[1,1], ny)
    zgrid = np.linspace(domain[2,0], domain[2,1], nz)
    tgrid = np.zeros((nt), dtype=float)
    for it in range(nt):
        tgrid[it] = it*dTout
    
    # create empty array for all solution data
    rho  = np.zeros((nx,ny,nz,nt), order='F')
    mx   = np.zeros((nx,ny,nz,nt), order='F')
    my   = np.zeros((nx,ny,nz,nt), order='F')
    mz   = np.zeros((nx,ny,nz,nt), order='F')
    et   = np.zeros((nx,ny,nz,nt), order='F')
    chem = np.zeros((nx,ny,nz,nchem,nt), order='F')
    
    # insert first processor's data into results array
    ist = subdomains[0,0]
    ind = subdomains[0,1]
    jst = subdomains[0,2]
    jnd = subdomains[0,3]
    kst = subdomains[0,4]
    knd = subdomains[0,5]
    nxl = ind-ist+1
    nyl = jnd-jst+1
    nzl = knd-kst+1
    for i in range(nt):
        rho[ist:ind+1,jst:jnd+1,kst:knd+1,i] = np.reshape(rho_data[i,:], (nxl,nyl,nzl), order='F')
        mx[ ist:ind+1,jst:jnd+1,kst:knd+1,i] = np.reshape( mx_data[i,:], (nxl,nyl,nzl), order='F')
        my[ ist:ind+1,jst:jnd+1,kst:knd+1,i] = np.reshape( my_data[i,:], (nxl,nyl,nzl), order='F')
        mz[ ist:ind+1,jst:jnd+1,kst:knd+1,i] = np.reshape( mz_data[i,:], (nxl,nyl,nzl), order='F')
        et[ ist:ind+1,jst:jnd+1,kst:knd+1,i] = np.reshape( et_data[i,:], (nxl,nyl,nzl), order='F')
        for ichem in range(nchem):
            chem[ist:ind+1,jst:jnd+1,kst:knd+1,ichem,i] = np.reshape(
                chem_data[ichem][i,:], (nxl,nyl,nzl), order='F')
        
    # iterate over remaining data files, inserting into output
    if (nprocs > 1):
        for isub in range(1,nprocs):
            rho_data = np.loadtxt('output-rho.' + repr(isub).zfill(7) + '.txt', dtype=np.double)
            mx_data  = np.loadtxt('output-mx.'  + repr(isub).zfill(7) + '.txt', dtype=np.double)
            my_data  = np.loadtxt('output-my.'  + repr(isub).zfill(7) + '.txt', dtype=np.double)
            mz_data  = np.loadtxt('output-mz.'  + repr(isub).zfill(7) + '.txt', dtype=np.double)
            et_data  = np.loadtxt('output-et.'  + repr(isub).zfill(7) + '.txt', dtype=np.double)
            for ichem in range(nchem):
                fname = 'output-c' + repr(ichem).zfill(cwidth) + '.' + repr(isub).zfill(7) + '.txt'
                chem_data[ichem] = np.loadtxt(fname,  dtype=np.double)
            # check that files have correct number of time steps
            if( (np.shape(rho_data)[0] != nt) or (np.shape(mx_data)[0] != nt) or (np.shape(my_data)[0] != nt) or
                (np.shape(mz_data)[0] != nt) or (np.shape(et_data)[0] != nt) ):
                sys.exit('error: an output for subdomain ' + repr(isub) + ' has an incorrect number of time steps')
            for ichem in range(nchem):
                if (np.shape(chem_data[ichem])[0] != nt):
                    sys.exit('error: an output for subdomain ' + repr(isub) + ' has an incorrect number of time steps')

            ist = subdomains[isub,0]
            ind = subdomains[isub,1]
            jst = subdomains[isub,2]
            jnd = subdomains[isub,3]
            kst = subdomains[isub,4]
            knd = subdomains[isub,5]
            nxl = ind-ist+1
            nyl = jnd-jst+1
            nzl = knd-kst+1
            for i in range(nt):
                rho[ist:ind+1,jst:jnd+1,kst:knd+1,i] = np.reshape(rho_data[i,:], (nxl,nyl,nzl), order='F')
                mx[ ist:ind+1,jst:jnd+1,kst:knd+1,i] = np.reshape( mx_data[i,:], (nxl,nyl,nzl), order='F')
                my[ ist:ind+1,jst:jnd+1,kst:knd+1,i] = np.reshape( my_data[i,:], (nxl,nyl,nzl), order='F')
                mz[ ist:ind+1,jst:jnd+1,kst:knd+1,i] = np.reshape( mz_data[i,:], (nxl,nyl,nzl), order='F')
                et[ ist:ind+1,jst:jnd+1,kst:knd+1,i] = np.reshape( et_data[i,:], (nxl,nyl,nzl), order='F')
                for ichem in range(nchem):
                    chem[ist:ind+1,jst:jnd+1,kst:knd+1,ichem,i] = np.reshape(
                        chem_data[ichem][i,:], (nxl,nyl,nzl), order='F')

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
