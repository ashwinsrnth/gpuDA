from gpuDA import *
import sys
from pycuda import autoinit

t1 = MPI.Wtime()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

npx = 3
npy = 3
npz = 3
    
assert(npx*npy*npz == size)

nx = 32
ny = 32
nz = 32

a = np.arange(nx*ny*nz, dtype=np.float64).reshape([nz, ny, nx])
# a_gpu = gpuarray.to_gpu(a)
# b_gpu = gpuarray.to_gpu(np.zeros([nz+2, ny+2, nx+2], dtype=np.float64))


comm = comm.Create_cart([npz, npy, npx], reorder=False)
da = GpuDA(comm, [nz, ny, nx], [npz, npy, npx], 1)

#da.halo_swap(a_gpu, b_gpu)

t2 = MPI.Wtime()
if rank == 0:
    print t2-t1

MPI.Finalize()
