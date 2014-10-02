from gpuDAtest import *
from pycuda import autoinit

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
proc_sizes = [3, 3, 3]
local_dims = [3, 3, 3]
nz, ny, nx = local_dims

da = create_test_da(proc_sizes, local_dims)

# fill a with rank
a = np.zeros([nz,ny,nx], dtype=np.float64)
a.fill(rank)
a_gpu = gpuarray.to_gpu(a)

# fill b with ones
b = np.ones([nz+2,ny+2,nx+2], dtype=np.float64)
b_gpu = gpuarray.to_gpu(b)

da.global_to_local(a_gpu, b_gpu)

# test that proper communication happened
if rank == 13:
    assert(np.all(da.left_recv_halo.get() == 12))
    assert(np.all(da.right_recv_halo.get() == 14))
    assert(np.all(da.bottom_recv_halo.get() == 10))
    assert(np.all(da.top_recv_halo.get() == 16))
    assert(np.all(da.front_recv_halo.get() == 4))
    assert(np.all(da.back_recv_halo.get() == 22))

# test that boundaries aren't affected by halo swap
if rank == 22:
    assert(np.all(b_gpu.get()[-1,:,:] == 1))

# fill b with rank
b = np.zeros([nz+2,ny+2,nx+2], dtype=np.float64)
b.fill(rank)
b_gpu = gpuarray.to_gpu(b)

# a is empty
a_gpu = gpuarray.empty([nz,ny,nx], dtype=np.float64)

da.local_to_global(b_gpu, a_gpu)

# test the ltog routines:
if rank == 13:
    assert(np.all(a_gpu.get() == 13))

if rank == 22:
    assert(np.all(b_gpu.get()[-1,:,:] == 22))

MPI.Finalize()
