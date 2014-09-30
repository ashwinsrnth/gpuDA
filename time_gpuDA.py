from gpuDA import *
import sys

t1 = MPI.Wtime()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

npx = 3
npy = 3
npz = 3
    
assert(npx*npy*npz == size)

nx = 2048
ny = 2048
nz = 3

a = np.arange(nx*ny*nz, dtype=np.float64).reshape([nz, ny, nx])
a_gpu = gpuarray.to_gpu(a)

comm = comm.Create_cart([npz, npy, npx], reorder=False)
da = GpuDA(comm, [nz, ny, nx], [npz, npy, npx], 1)
da.halo_swap(a_gpu)

t2 = MPI.Wtime()

if rank == 0:
    print t2-t1

MPI.Finalize()
