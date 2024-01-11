import numpy as np
import time
import gc
import sys
import h5py
import yaml
import os
from common_functions import get_memory, kroneckerdelta
import h5py
import numpy as np
import pyfftw
import Pk_library as PKL
import gc
from common_functions import readGadgetSnapshot
from mpi4py import MPI
import h5py
import glob
import MAS_library as MASL
import random
from concurrent.futures import ThreadPoolExecutor
import os



def delta_to_tidesq(delta_k, nmesh, lbox, nthread, posarr, rank, comm):
    '''
    Computes the square tidal field from the density FFT

    s^2 = s_ij s_ij

    where

    s_ij = (k_i k_j / k^2 - delta_ij / 3 ) * delta_k

    Inputs:
    delta_k: fft'd density, slab-decomposed.
    nmesh: size of the mesh
    lbox: size of the box
    rank: current MPI rank
    nranks: total number of MPI ranks

    Outputs:
    tidesq: the s^2 field for the given slab.
    '''
    nranks = comm.Get_size()
    kvals = np.fft.fftfreq(nmesh)*(2*np.pi*nmesh)/lbox
    kvalsmpi = kvals[rank*nmesh//nranks:(rank+1)*nmesh//nranks]
    kvalsr = np.fft.rfftfreq(nmesh)*(2*np.pi*nmesh)/lbox
    ky, kx,  kz = np.meshgrid(kvals, kvalsmpi, kvalsr, sparse=True)
    print(kx.shape, delta_k.shape, ky.shape, kz.shape)
    knorm = kx**2 + ky**2 + kz**2
    if knorm[0][0][0] == 0:
        knorm[0][0][0] = 1

    klist = [['x','x'],['x','y'],['x','z'],['y','y'],['y', 'z'], ['z', 'z']]
    gc.collect()

    #Compute the symmetric tide at every Fourier mode which we'll reshape later

    #Order is xx, xy, xz, yy, yz, zz
    jvec = [[0,0], [0,1], [0,2], [1,1], [1,2], [2,2]]
    tidesq = np.zeros(posarr.shape[0], dtype=np.float32)
    itemsize = MPI.DOUBLE_COMPLEX.Get_size()
    size = nmesh*nmesh*len(kvalsr)
    if rank==0:
        nbytes = itemsize*size
    else:
        nbytes = 0
    shape = (nmesh, nmesh, len(kvalsr))

    for i in range(len(klist)):
        print("{0} k, {1}".format(i, len(klist)), flush=True)
        if((klist[i][0]=="x") and (klist[i][1]=="x")):
            fft_tide = np.array((kx*kx/knorm - kroneckerdelta(jvec[i][0], jvec[i][1])/3.), dtype='complex64')*delta_k
        elif((klist[i][0]=="x") and (klist[i][1]=="y")):
            fft_tide = np.array((kx*ky/knorm - kroneckerdelta(jvec[i][0], jvec[i][1])/3.), dtype='complex64')*delta_k
        elif((klist[i][0]=="x") and (klist[i][1]=="z")):
            fft_tide = np.array((kx*kz/knorm - kroneckerdelta(jvec[i][0], jvec[i][1])/3.), dtype='complex64')*delta_k
        elif((klist[i][0]=="y") and (klist[i][1]=="y")):
            fft_tide = np.array((ky*ky/knorm - kroneckerdelta(jvec[i][0], jvec[i][1])/3.), dtype='complex64')*delta_k
        elif((klist[i][0]=="y") and (klist[i][1]=="z")):
            fft_tide = np.array((ky*kz/knorm - kroneckerdelta(jvec[i][0], jvec[i][1])/3.), dtype='complex64')*delta_k
        elif((klist[i][0]=="z") and (klist[i][1]=="z")):
            fft_tide = np.array((kz*kz/knorm - kroneckerdelta(jvec[i][0], jvec[i][1])/3.), dtype='complex64')*delta_k
        else:
            assert(0)
        comm.Barrier()
        win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm) 
        buf, itemsize = win.Shared_query(0)  
        assert itemsize == MPI.DOUBLE_COMPLEX.Get_size() 
        ary = np.ndarray(buffer=buf, dtype=np.complex64, shape=(size,)) 
        ary[(rank*nmesh//nranks)*shape[1]*shape[2]:((rank+1)*nmesh//nranks)*shape[1]*shape[2]] = fft_tide.ravel() 
        comm.Barrier() 
        fft_tide = ary.reshape(shape)
        if rank==0:
            real_out = PKL.IFFT3Dr_f(fft_tide, nthread)
        else:
            real_out =None
        print("begin real_out Deltas, Deltas field", flush=True)
        if rank==0:
            print(real_out)
        real_out, win_in = mpishapememory(MPI.FLOAT, np.float32, real_out, rank, comm)
        print("done real_out Deltas field", flush=True)
        w = np.zeros(len(posarr), dtype=np.float32)
        MASL.CIC_interp(real_out, lbox, posarr, w)
        tidesq += 1.*w**2
        if jvec[i][0] != jvec[i][1]:
            tidesq+= 1.*w**2
        comm.Barrier()
        del fft_tide
        del w
        del real_out
        win.Free()
        win_in.Free()
        gc.collect()
        #kx, ky, kz = np.meshgrid(kvals,kvals,  kvalsr)
        #knorm = kx**2 + ky**2 + kz**2
        #if knorm[0][0][0] == 0:
        #    knorm[0][0][0] = 1

    # pass
    del kx,ky,kz
    gc.collect()

    return tidesq

def delta_to_gradsqdelta(delta_k, nmesh, lbox, nthread):
    '''
    Computes the density curvature from the density FFT

    nabla^2 delta = IFFT(-k^2 delta_k)

    Inputs:
    delta_k: fft'd density, slab-decomposed.
    nmesh: size of the mesh
    lbox: size of the box
    rank: current MPI rank
    nranks: total number of MPI ranks
    fft: PFFT fourier transform object. Used to do the backwards FFT.

    Outputs:
    real_gradsqdelta: the nabla^2delta field for the given slab.
    '''

    kvals = np.fft.fftfreq(nmesh)*(2*np.pi*nmesh)/lbox
    kvalsr = np.fft.rfftfreq(nmesh)*(2*np.pi*nmesh)/lbox

    kx, ky, kz = np.meshgrid(kvals,kvals,  kvalsr)


    knorm = kx**2 + ky**2 + kz**2
    if knorm[0][0][0] == 0:
        knorm[0][0][0] = 1

    del kx, ky, kz
    gc.collect()

    #Compute -k^2 delta which is the gradient
    ksqdelta = -np.array(knorm * (delta_k), dtype='complex64')

    real_gradsqdelta = IFFT3Dr_f(ksqdelta, nthread)
    return real_gradsqdelta

# This function performs the 3D FFT of a field in double precision
def IFFT3Dr_f(a,threads):
    # align arrays
    dims  = len(a)
    a_in  = pyfftw.empty_aligned((dims,dims,dims//2+1),dtype='complex64')
    a_out = pyfftw.empty_aligned((dims,dims,dims),    dtype='float32')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=(0,1,2),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_BACKWARD', threads=threads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a;  fftw_plan(a_in,a_out);  
    del a_in;
    del fftw_plan;
    gc.collect()
    return a_out
def FFT3Dr_f( a,  threads):
    # align arrays
    dims  = len(a)
    a_in  = pyfftw.empty_aligned((dims,dims,dims),    dtype='float32')
    a_out = pyfftw.empty_aligned((dims,dims,dims//2+1),dtype='complex64')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=(0,1,2),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_FORWARD', threads=threads)
    # put input array into delta_r and perform FFTW
    a_in [:] = a;  fftw_plan(a_in,a_out);  
    del a_in;
    del fftw_plan;
    gc.collect()
    return a_out
def savedata(lists):
    global outname
    np.save(outname+".{0}".format(lists[0]), savarr[lists[1]:lists[2]])



def CIC_interp(density, BoxSize, pos):
    print("doing CIC__3", flush=True)
    # find number of particles, the inverse of the cell size and dims
    den = np.zeros(pos.shape[0])
    particles = pos.shape[0];  dims = density.shape[0]
    inv_cell_size = dims/BoxSize
    u = np.zeros((pos.shape[0],3), dtype=np.float32)
    d = np.zeros((pos.shape[0],3), dtype=np.float32)
    index_d = np.zeros((pos.shape[0],3), dtype=np.int64)
    index_u= np.zeros((pos.shape[0],3), dtype=np.int64)
    # do a loop over all particles
       
    dist          = pos*inv_cell_size
    u    = dist - dist.astype(int)
    d      = 1.0 - u[:]
    index_d = ((dist).astype(int)%dims)
    index_u = index_d[:] + 1
    index_u = (index_u[:]%dims) #seems this is faster
    print("doing CIC__4", flush=True)
    den = density[index_d[:,0],index_d[:,1],index_d[:,2]]*d[:,0]*d[:,1]*d[:,2]
    print("doing CIC__5", flush=True)
    den += density[index_d[:,0],index_d[:,1],index_u[:,2]]*d[:,0]*d[:,1]*u[:,2]+\
             density[index_d[:,0],index_u[:,1],index_d[:,2]]*d[:,0]*u[:,1]*d[:,2]+\
             density[index_d[:,0],index_u[:,1],index_u[:,2]]*d[:,0]*u[:,1]*u[:,2]+\
             density[index_u[:,0],index_d[:,1],index_d[:,2]]*u[:,0]*d[:,1]*d[:,2]+\
             density[index_u[:,0],index_d[:,1],index_u[:,2]]*u[:,0]*d[:,1]*u[:,2]+\
             density[index_u[:,0],index_u[:,1],index_d[:,2]]*u[:,0]*u[:,1]*d[:,2]+\
             density[index_u[:,0],index_u[:,1],index_u[:,2]]*u[:,0]*u[:,1]*u[:,2]
    return den
import cic
def parallel_CIC_inter(lists):
    print("doing CIC_{0}".format(lists[0]), flush=True)
    global CICdata
    delta_noiseless, lbox, posarr = CICdata
    print("doing CIC_{0}2".format(lists[0]), flush=True)
    w = np.zeros(posarr[lists[1]:lists[2]].shape[0], dtype=np.float32)
    cic.CIC_interp(delta_noiseless, lbox, posarr[lists[1]:lists[2]], w)
    #w = CIC_interp(delta_noiseless, lbox, posarr[lists[1]:lists[2]])
    print("doing CIC_{0} done".format(lists[0]), flush=True)
    return w
def mpishapememory(MPIdtype, dtype, data, rank, comm):
    itemsize =  MPIdtype.Get_size()
    if rank==0:
        delta_noiseless = data 
        size = np.prod(delta_noiseless.shape)
        shape = delta_noiseless.shape
        nbytes = itemsize*size
    else:
        delta_noiseless = None
        nbytes=0
        size=None
        shape = None
    comm.Barrier()
    size = comm.bcast(size, root=0)
    shape = comm.bcast(shape, root=0)
    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)
    buf, itemsize = win.Shared_query(0)
    assert itemsize == MPIdtype.Get_size()
    ary = np.ndarray(buffer=buf, dtype=dtype, shape=(size,))
    if rank == 0:
        ary[:] = delta_noiseless.ravel() 
    comm.Barrier()
    delta_noiseless = ary.reshape(shape)
    return delta_noiseless, win


if __name__ == "__main__":
    yamldir = sys.argv[1]
    configs = yaml.safe_load(open(yamldir, 'r'))
    fdir = configs['particledir']

    lindir = configs['outdir']
    nmesh = configs['nmesh_in'] 
    start_time = time.time()
    lbox = configs['lbox']
    outdir = lindir
    dodeltafield=True
    dodeltasqfield=True
    dodeltasfield=True
    nablasqfield=False
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()
    assert(configs['nparticle']%nranks==0)
    posname = lindir+"pos_{0}_{1}.npy".format(rank, nranks)
    if os.path.isfile(posname):
        posarr = np.load(posname)
    else:
        glass = readGadgetSnapshot(configs['glass'],print_header=True,read_pos=True, read_vel=False,read_id=False, single_type=1) 
        TileFac = configs['TileFac']
        lenpos = len(glass[1])
        Box=lbox*1000
        def IDmapping(ID):
            length = ID-1
            i = (length%TileFac)
            length = (length-i)/TileFac
            j = length%TileFac
            length = (length-j)/TileFac
            m = (length%lenpos).astype(int)
            k = (length-m)/lenpos
            x,y,z = (glass[1][m,0]/TileFac+ k * (Box / TileFac)),(glass[1][m,1]/TileFac+ j * (Box / TileFac)),(glass[1][m,2]/TileFac+ i* (Box / TileFac))
            return x/1E3,y/1E3,z/1E3
        #idvec=np.arange(1, configs['nparticle']+1)
        high = int(np.ceil(configs['nparticle']/nranks))*(rank+1)
        low = int(np.ceil(configs['nparticle']/nranks))*(rank)
        idvec = np.arange(low+1,high+1)
        x_ic, y_ic, z_ic = IDmapping(idvec)
        posarr = np.array([x_ic, y_ic, z_ic],dtype=np.float32).T
        del x_ic
        del y_ic
        del z_ic
        del idvec
        del glass
        gc.collect()
        np.save(posname, posarr)
    itemsize = MPI.FLOAT.Get_size() 
    if rank==0:
        delta_noiseless = np.load(lindir+'linICfield.npy', 'r')
        size = np.prod(delta_noiseless.shape)
        shape = delta_noiseless.shape
        nbytes = itemsize*size
    else:
        delta_noiseless = None
        nbytes=0
        size=None
        shape = None
    comm.Barrier()
    size = comm.bcast(size, root=0)
    shape = comm.bcast(shape, root=0)
    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)
    buf, itemsize = win.Shared_query(0)
    assert itemsize == MPI.FLOAT.Get_size()
    ary = np.ndarray(buffer=buf, dtype=np.float32, shape=(size,))
    if rank == 0:
        ary[:] = delta_noiseless.ravel() 
    comm.Barrier()
    delta_noiseless = ary.reshape(shape)

    #Delta field
    if dodeltafield:
        outname = outdir + "delta_np.npy"
        print("do Delta field", flush=True)
        w = np.zeros(len(posarr), dtype=np.float32)
        MASL.CIC_interp(delta_noiseless, lbox, posarr, w)
        np.save(outname+".{0}".format(rank), w)

    if dodeltasqfield:
        print("do Deltasq field", flush=True)
        deltasqchto = w**2 
        deltasqchto -= np.mean(deltasqchto)
        outname = outdir + "deltasq_np.npy" 
        np.save(outname+".{0}".format(rank), deltasqchto)
        del w
        del deltasqchto
        gc.collect()
    
    #Delta_s
    if dodeltasfield:
        print("do Deltas field", flush=True)
        if rank == 0:
            delta_k = PKL.FFT3Dr_f(delta_noiseless,64)
        else:
            delta_k = None
        comm.Barrier()
        del delta_noiseless
        win.Free()
        gc.collect()
        delta_k, win_in = mpishapememory(MPI.DOUBLE_COMPLEX, np.complex64, delta_k, rank, comm)
        print("done delta_k Deltas field", flush=True)
        tidesq_chto = delta_to_tidesq(delta_k[rank*nmesh//nranks:(rank+1)*nmesh//nranks], nmesh, lbox, 64, posarr, rank, comm)
        tidesq_chto = tidesq_chto-np.mean(tidesq_chto)
        savarr = tidesq_chto
        outname = outdir + "tidesq_np.npy" 
        np.save(outname+".{0}".format(rank), tidesq_chto)
        del tidesq_chto 
        del delta_k
        win_in.Free()
