from gpuDAtest import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
proc_sizes = [1, 1, 3]
local_dims = [1, 3, 3]

da, a_gpu, b_gpu = setup_test(proc_sizes, local_dims)

# test center:
if rank == 1:
    assert(np.all(da.left_recv_halo.get() == 0))
    assert(np.all(da.left_send_halo.get() == 1))
    assert(np.all(da.right_send_halo.get() == 1))
    assert(np.all(da.right_recv_halo.get() == 2))

# test left and right:
if rank == 0:
    assert(np.all(da.right_recv_halo.get() == 1))

if rank == 2:
    assert(np.all(da.left_recv_halo.get() == 1))

if rank == 1:
    print b_gpu

MPI.Finalize()
