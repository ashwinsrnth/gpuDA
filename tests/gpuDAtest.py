import nose
import sys
sys.path.append('..')

from gpuDA import *

def setup_test(proc_sizes, local_dims):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    npz, npy, npx = proc_sizes    
    nz, ny, nx = local_dims
    assert(npx*npy*npz == size)

    a = np.zeros([nz,ny,nx],dtype=np.float64)
    a.fill(rank)
    a_gpu = gpuarray.to_gpu(a)

    comm = comm.Create_cart([npz, npy, npx], reorder=False)
    da = GpuDA(comm, [nz, ny, nx], [npz, npy, npx], 1)
    da.halo_swap(a_gpu)

    return da
