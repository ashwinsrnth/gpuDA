import sys
sys.path.append('..')
from gpuDA import *


def create_test_da(proc_sizes, local_dims):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    npz, npy, npx = proc_sizes    
    nz, ny, nx = local_dims
    assert(npx*npy*npz == size)
  
    comm = comm.Create_cart([npz, npy, npx], reorder=True)
    da = GpuDA(comm, [nz, ny, nx], [npz, npy, npx], 1)
     
    return da
