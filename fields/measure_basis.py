import pmesh
import numpy as np
import time
#from nbodykit.lab import FieldMesh
from nbodykit.algorithms.fftpower import FFTPower
# from nbodykit.algorithms.fftcorr import FFTCorr
#from nbodykit.source.mesh import ArrayMesh
import sys
import gc
import pyccl
import pandas as pd
import psutil
import os
import yaml
from mpi4py import MPI
from common_functions import readGadgetSnapshot
import h5py
import glob

def get_memory(rank):
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss/1e9, "GB is current memory usage, rank ", rank)  # in bytes 


def mpiprint(text):
    if rank==0:
        print(text)
        sys.stdout.flush()
    else:
        pass
def CompensateCICAliasing(w, v):
    """
    Return the Fourier-space kernel that accounts for the convolution of
        the gridded field with the CIC window function in configuration space,
            as well as the approximate aliasing correction
    From the nbodykit documentation.
    """
    for i in range(3):
        wi = w[i]
        v = v / (1 - 2. / 3 * np.sin(0.5 * wi) ** 2) ** 0.5
    return v


################################### YAML /Initial Config stuff #################################
yamldir = sys.argv[1]


##PALLIATIVE 
fieldnameadd = sys.argv[2]


configs = yaml.load(open(yamldir, 'r'), yaml.FullLoader)

lindir = configs['outdir']

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()

nmesh = configs['nmesh_in'] 


bigarr = []
start_time = time.time()
Lbox = configs['lbox']

compensate = bool(configs['compensate'])


# fdir = '/oak/stanford/orgs/kipac/aemulus/aemulus_alpha/Box%03d/output/snapdir_%03d/snapshot_%03d.'%(boxno, snapdir, snapdir)
fdir = configs['particledir']

#Save to wherever particles are
componentdir = configs['outdir']
boxno = configs['aem_box']
try:
    testvar = configs['aem_testno']
except:
    testvar = ''

start_time = time.time()

################################################################################################
#################################### Advecting weights #########################################
# componentdir = '/home/users/kokron/scratch/ptbias_emu/Box%03d/snapdir_%03d/'%(boxno, snapdir)

# comps = glob(componentdir+'componentfields_*')


#print(growthratio)

# N = 5


pm = pmesh.pm.ParticleMesh([nmesh, nmesh, nmesh], Lbox, dtype='float32', resampler='cic', comm=comm)


#ParticleMeshes for 1, delta, deltasq, tidesq, nablasq
#Make the late-time component fields
fieldlist = [pm.create(type='real'),pm.create(type='real'),pm.create(type='real'),pm.create(type='real'),pm.create(type='real')]
if rank==0:
    get_memory(rank)
    print('starting loop')
    sys.stdout.flush()



# #Load in component field for this part of fdir
# compfield = np.fromfile(componentdir+'componentfields_%s.npy'%rank,dtype='float32')
# compfield = compfield.reshape(len(compfield)//4, 4)
# mpiprint('loaded compfield og')

# randcat = np.zeros(shape=(len(compfield), 3), dtype='float32')



# for i in range(4):
#     np.save(componentdir+'reshape_componentfields_%s_%s.npy'%(rank,i+1), compfield[:,i])
# del compfield
# gc.collect()

lenrand = 0

#Load in a subset of the total gadget snapshot. 
#TODO: this is hard-coded for Sherlock and Aemulus but should change for generic N-body sims.
if configs['sim_type']=="Gadget":
    for i in range(16*rank, 16*(rank+1)):
        gadgetsnap = readGadgetSnapshot(fdir+'%s'%i, read_id=True, read_pos=True)

        gadgetpos = gadgetsnap[1]

        gadgetidx = gadgetsnap[2]
        if i == 16*rank:
            posvec = 1.*gadgetpos
            idvec = 1.*gadgetidx
            mpiprint('here!')
            mpiprint(posvec.shape)
        else:
            posvec = np.vstack((posvec, gadgetpos))
            idvec = np.hstack((idvec, gadgetidx))
        lenrand+=len(gadgetpos)
        del gadgetsnap
        gc.collect()


elif configs['sim_type']=="illustris":
    if "downsamplefactor" in configs.keys():
       downsamplefactor = configs["downsamplefactor"]
    else:
        downsamplefactor = 1
    filesname = glob.glob(fdir+"*")
    split = np.array_split(np.array(filesname), nranks)
    i = rank
    for j, fin in enumerate(split[i]):
        infoma = h5py.File(fin)
        gadgetpos = 1.*infoma['PartType1/Coordinates'][::downsamplefactor]/1E3
        gadgetidx = 1*infoma['PartType1/ParticleIDs'][::downsamplefactor] 
        if j==0:
            posvec = gadgetpos
            idvec = gadgetidx 
        else:
            posvec = np.vstack((posvec, gadgetpos))
            idvec = np.hstack((idvec, gadgetidx))
        mpiprint('here i={0}, j={1}!'.format(i,j))
        mpiprint(posvec.shape)
    box_scale= 1/(infoma['Header'].attrs['Redshift']+1)
    mpiprint(box_scale) 
else:
    raise NotImplementedError("{0} not implemented ".format(configs['sim_type']))



#Gadget has IDs starting with ID=1. 
#FastPM has ID=0
#idfac decides which one to use
idfac = 1
if (configs['sim_type']=="Gadget") | (configs['sim_type'] == 'FastPM'):
    if configs['sim_type'] == 'FastPM':
        idfac = 0

    a_ic = ((idvec-idfac)//nmesh**2)%nmesh
    b_ic = ((idvec-idfac)//nmesh)%nmesh
    c_ic = (idvec-idfac)%nmesh
    mpiprint(a_ic[3])
    a_ic = a_ic.astype(int)
    b_ic = b_ic.astype(int)
    c_ic = c_ic.astype(int)
elif configs['sim_type']=="illustris":
    glass = readGadgetSnapshot(configs['glass'],print_header=True,read_pos=True, read_vel=False,read_id=False, single_type=1) 
    TileFac = configs['TileFac']
    lenpos = len(glass[1])
    Box=Lbox*1000
    def IDmapping(ID):
        length = ID-1
        i = (length%TileFac)
        length = (length-i)/TileFac
        j = length%TileFac
        length = (length-j)/TileFac
        m = (length%lenpos).astype(int)
        k = (length-m)/lenpos
        x,y,z = (glass[1][m,0]/TileFac+ k * (Box / TileFac)),(glass[1][m,1]/TileFac+ j * (Box / TileFac)),(glass[1][m,2]/TileFac+ i* (Box / TileFac))
        return x,y,z
    mpiprint("mapping ID")
    x_ic, y_ic, z_ic = IDmapping(idvec)
    meshlength = Box/nmesh
    a_ic = (x_ic/meshlength).astype(int)%nmesh
    b_ic = (y_ic/meshlength).astype(int)%nmesh
    c_ic = (z_ic/meshlength).astype(int)%nmesh
    
else:
    raise NotImplementedError("{0} not implemented ".format(configs['sim_type']))
#Figure out where each particle position is going to be distributed among mpi ranks
layout = pm.decompose(posvec)

#Exchange positions
p = layout.exchange(posvec)

mpiprint(('posvec shapes', posvec.shape))

mpiprint(('idvec shapes', idvec.shape))
del posvec
gc.collect()



# f = h5py.File(lindir+'mpi_icfields_nmesh%s.h5'%nmesh, 'r')
# keynames = list(f.keys())
keynames = ['1', 'delta', 'deltasq', 'tidesq', 'nablasq']
for k in range(len(fieldlist)):
    if rank==0:
        print(k)
    if k == 0:
        pm.paint(p, out=fieldlist[k], mass = 1, resampler='cic')
    else:
        #Now only load specific compfield. 1,2,3 is delta, delta^2, s^2

        # compfield = np.load(componentdir+'reshape_componentfields_%s_%s.npy'%(rank,k), mmap_mode='r')
        #Load in the given weight field
        arr = np.load(lindir+keynames[k]+'_np.npy', mmap_mode='r')
       
        #Get weights
        w = arr[a_ic, b_ic, c_ic]


        mpiprint(('w shapes', w.shape))

        #distribute weights properly
        m = layout.exchange(w)


        del w
        gc.collect()

        get_memory(rank)

        pm.paint(p, out=fieldlist[k], mass = m, resampler='cic')
        sys.stdout.flush()
        del m
        gc.collect()

    #print('painted! ', rank)
    sys.stdout.flush()
if rank==0:
    print(fieldlist[0].shape)
del p
gc.collect()
if rank==0:
    print('pasted')
    sys.stdout.flush()
get_memory(rank)

#Normalize and mean-subtract the normal aprticle field.
fieldlist[0] = fieldlist[0]/fieldlist[0].cmean() - 1
for k in range(len(fieldlist)):
    if rank==0:
        print(np.mean(fieldlist[k].value), np.std(fieldlist[k].value))
        sys.stdout.flush()
    np.save(componentdir+'latetime_weight_%s_%s_%s_rank%s'%(k,nmesh,fieldnameadd,rank), fieldlist[k].value)
    if compensate:
        fieldlist[k] = fieldlist[k].r2c()
        fieldlist[k] = fieldlist[k].apply(CompensateCICAliasing, kind='circular')

get_memory(rank)
sys.stdout.flush()

#######################################################################################################################
#################################### Adjusting for growth #############################################################

field_dict = {'1': fieldlist[0], r'$\delta_L$': fieldlist[1], r'$\delta^2$': fieldlist[2], r'$s^2$':fieldlist[3], r'$\nabla^2\delta$':fieldlist[4]}
labelvec = ['1',r'$\delta_L$',  r'$\delta^2$',  r'$s^2$',r'$\nabla^2\delta$']


#Get the box redshift and cosmology to compute the growth factor. 
#This is currently hard-coded to work with AEMULUS only. Make more general
if (configs['sim_type']=="Gadget") | (configs['sim_type'] == 'FastPM'):
    box_scale = readGadgetSnapshot(fdir+'0')[2]
zbox = 1./box_scale - 1
#Get growth factor 
if (configs['sim_type']=="Gadget") | (configs['sim_type'] == 'FastPM'):
    if 'Test' in fdir:
        cosmofiles = pd.read_csv('/home/users/swmclau2/Git/pearce/pearce/mocks/test_cosmos.txt', sep=' ')
        boxcosmo = cosmofiles.iloc[boxno]
    else:
        cosmofiles = pd.read_csv('/home/users/kokron/Projects/lakelag/cosmos.txt', sep=' ')
        boxcosmo = cosmofiles.iloc[boxno]
    cosmo = pyccl.Cosmology(Omega_b= boxcosmo['ombh2']/(boxcosmo['H0']/100)**2, Omega_c = boxcosmo['omch2']/(boxcosmo['H0']/100)**2, h = boxcosmo['H0']/100, n_s = boxcosmo['ns'], w0=boxcosmo['w0'], Neff=boxcosmo['Neff'],sigma8 = boxcosmo['sigma8'])

elif configs['sim_type']=="illustris":
    boxcosmo =  configs['cosmo']
    cosmo = pyccl.Cosmology(Omega_b= boxcosmo['ombh2']/(boxcosmo['H0']/100)**2, Omega_c = boxcosmo['omch2']/(boxcosmo['H0']/100)**2, h = boxcosmo['H0']/100, n_s = boxcosmo['ns'], w0=boxcosmo['w0'], Neff=boxcosmo['Neff'],sigma8 = boxcosmo['sigma8'])

else:
    raise NotImplementedError("{0} not implemented ".format(configs['sim_type']))
    



#Aemulus boxes have IC at z=49
z_ic=configs['z_ic']
#Compute relative growth from IC to snapdir 
growthratio = pyccl.growth_factor(cosmo, [box_scale])/pyccl.growth_factor(cosmo, 1./(1+z_ic))
#Vector to rescale component spectra with appropriate linear growth factors.
D = growthratio
mpiprint(D)
#If not including nabla field
#growthratvec = np.array([1, D, D**2, D**2, D**3, D**4, D**2, D**3, D**4, D**4])

growthratvec = np.array([1, D, D**2, D**2, D**3, D**4, D**2, D**3, D**4, D**4,
                        D, D**2, D**3, D**3, D**2])
#######################################################################################################################
#################################### Measuring P(k) ###################################################################
kpkvec = []
rxivec = []
pkcounter = 0
for i in range(5):
    for j in range(5):
        if i<j:
            pass
        if i>=j:
            pk = FFTPower(field_dict[labelvec[i]], '1d', second = field_dict[labelvec[j]], BoxSize=Lbox, Nmesh=nmesh)

            #The xi measurements don't work great for now.
            #xi = FFTCorr(field_dict[labelvec[i]], '1d', second = field_dict[labelvec[j]], BoxSize=Lbox, Nmesh=nmesh)
            pkpower = pk.power['power'].real
            #xicorr = xi.corr['corr'].real
            if i==0 and j==0:
                kpkvec.append(pk.power['k'])
            #    rxivec.append(xi.corr['r'])
            kpkvec.append(pkpower*growthratvec[pkcounter])
            #rxivec.append(xicorr*growthratvec[pkcounter])
            pkcounter+=1
            mpiprint(('pk done ', pkcounter))



kpkvec = np.array(kpkvec)
#rxivec = np.array(rxivec)
if rank==0:
    np.savetxt(componentdir+'lakelag_mpi_pk_box%s_a%.2f_nmesh%s.txt'%(boxno,box_scale,nmesh), kpkvec)
    #np.savetxt(componentdir+'lakelag_mpi_xi_box%s_snap%s_nmesh%s.txt'%(boxno,snapdir,nmesh), rxivec)
    # os.system('rm -r '+componentdir+'reshape_componentfields_*')
    #os.system('rm -r '+componentdir+'componentfields_*')
    
    print(time.time() - start_time)
    # for k in range(compfield.shape[1] + 1):
    #     if n == 0:
    #         layout = pm.decompose(randcat)

    #         p = layout.exchange(randcat)
            
    #         fieldlist[k] = RealField(pm)
    #         test = pm.paint(p, out=fieldlist[k], mass = compfield[:,k], resampler='cic')

    #     else:
    #         layout = pm.decompose(randcat)

    #         p = layout.exchange(randcat)
            
    #         field_update = RealField(pm)
            
    #         test = pm.paint(p, out=field_update, mass = compfield[:,k], resampler='cic')
            
    #         fieldlist[k] += field_update
