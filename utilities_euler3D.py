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
        sname = 'output-euler3D_subdomain.' + repr(i).zfill(7) + '.txt'
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
    sname = 'output-euler3D_subdomain.0000000.txt'
    subd = np.loadtxt(sname, dtype=np.int)
    nx = subd[0]
    ny = subd[1]
    nz = subd[2]
    subdomains[0,:] = subd[3:]
    for i in range(1,nprocs):
        sname = 'output-euler3D_subdomain.' + repr(i).zfill(7) + '.txt'
        subd = np.loadtxt(sname, dtype=np.int)
        if ((subd[0] != nx) or (subd[1] != ny) or (subd[2] != nz)):
            sys.exit("error: subdomain files incompatible (clean up and re-run test)")
        subdomains[i,:] = subd[3:]
    return [nx, ny, nz, subdomains]

# load solution data into multi-dimensional arrays
def load_data():

    # load metadata
    nprocs = get_MPI_tasks()
    nx, ny, nz, subdomains = load_subdomain_info(nprocs)
    
    # load first processor's data, and determine total number of time steps
    rho_data = np.loadtxt('output-euler3D_rho.0000000.txt', dtype=np.double)
    mx_data  = np.loadtxt('output-euler3D_mx.0000000.txt',  dtype=np.double)
    my_data  = np.loadtxt('output-euler3D_my.0000000.txt',  dtype=np.double)
    mz_data  = np.loadtxt('output-euler3D_mz.0000000.txt',  dtype=np.double)
    et_data  = np.loadtxt('output-euler3D_et.0000000.txt',  dtype=np.double)
    nt = np.shape(rho_data)[0]
    if ( (np.shape(mx_data)[0] != nt) or (np.shape(my_data)[0] != nt) or
         (np.shape(mz_data)[0] != nt) or (np.shape(et_data)[0] != nt) ):
        sys.exit('error: an output for subdomain 0 has an incorrect number of time steps')
    
    
    # create empty array for all solution data
    rho = np.zeros((nx,ny,nz,nt), order='F')
    mx  = np.zeros((nx,ny,nz,nt), order='F')
    my  = np.zeros((nx,ny,nz,nt), order='F')
    mz  = np.zeros((nx,ny,nz,nt), order='F')
    et  = np.zeros((nx,ny,nz,nt), order='F')

    # insert first processor's data into results array
    istart = subdomains[0,0]
    iend = subdomains[0,1]
    jstart = subdomains[0,2]
    jend = subdomains[0,3]
    kstart = subdomains[0,4]
    kend = subdomains[0,5]
    nxl = iend-istart+1
    nyl = jend-jstart+1
    nzl = kend-kstart+1
    for i in range(nt):
        rho[istart:iend+1,jstart:jend+1,kstart:kend+1,i] = np.reshape(rho_data[i,:], (nxl,nyl,nzl), order='F')
        mx[ istart:iend+1,jstart:jend+1,kstart:kend+1,i] = np.reshape( mx_data[i,:], (nxl,nyl,nzl), order='F')
        my[ istart:iend+1,jstart:jend+1,kstart:kend+1,i] = np.reshape( my_data[i,:], (nxl,nyl,nzl), order='F')
        mz[ istart:iend+1,jstart:jend+1,kstart:kend+1,i] = np.reshape( mz_data[i,:], (nxl,nyl,nzl), order='F')
        et[ istart:iend+1,jstart:jend+1,kstart:kend+1,i] = np.reshape( et_data[i,:], (nxl,nyl,nzl), order='F')
    
    # iterate over remaining data files, inserting into output
    if (nprocs > 1):
        for isub in range(1,nprocs):
            rho_data = np.loadtxt('output-euler3D_rho.' + repr(isub).zfill(7) + '.txt', dtype=np.double)
            mx_data  = np.loadtxt('output-euler3D_mx.'  + repr(isub).zfill(7) + '.txt', dtype=np.double)
            my_data  = np.loadtxt('output-euler3D_my.'  + repr(isub).zfill(7) + '.txt', dtype=np.double)
            mz_data  = np.loadtxt('output-euler3D_mz.'  + repr(isub).zfill(7) + '.txt', dtype=np.double)
            et_data  = np.loadtxt('output-euler3D_et.'  + repr(isub).zfill(7) + '.txt', dtype=np.double)
            # check that files have correct number of time steps
            if( (np.shape(rho_data)[0] != nt) or (np.shape(mx_data)[0] != nt) or (np.shape(my_data)[0] != nt) or
                (np.shape(mz_data)[0] != nt) or (np.shape(et_data)[0] != nt) ):
                sys.exit('error: an output for subdomain ' + isub + ' has an incorrect number of time steps')
            istart = subdomains[isub,0]
            iend = subdomains[isub,1]
            jstart = subdomains[isub,2]
            jend = subdomains[isub,3]
            kstart = subdomains[isub,4]
            kend = subdomains[isub,5]
            nxl = iend-istart+1
            nyl = jend-jstart+1
            nzl = kend-kstart+1
            for i in range(nt):
                rho[istart:iend+1,jstart:jend+1,kstart:kend+1,i] = np.reshape(rho_data[i,:], (nxl,nyl,nzl), order='F')
                mx[ istart:iend+1,jstart:jend+1,kstart:kend+1,i] = np.reshape(rho_data[i,:], (nxl,nyl,nzl), order='F')
                my[ istart:iend+1,jstart:jend+1,kstart:kend+1,i] = np.reshape(rho_data[i,:], (nxl,nyl,nzl), order='F')
                mz[ istart:iend+1,jstart:jend+1,kstart:kend+1,i] = np.reshape(rho_data[i,:], (nxl,nyl,nzl), order='F')
                et[ istart:iend+1,jstart:jend+1,kstart:kend+1,i] = np.reshape(rho_data[i,:], (nxl,nyl,nzl), order='F')

    return [nx, ny, nz, nt, rho, mx, my, mz, et]



##### end of script #####
