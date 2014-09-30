from gpuDAtest import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
proc_sizes = [3, 3, 3]
local_dims = [3, 3, 3]

da, a_gpu, b_gpu = setup_test(proc_sizes, local_dims)

# test that the center received the right
# values from all 6 neighbours:
if rank == 13:
    assert(np.all(da.left_recv_halo.get() == 12))
    assert(np.all(da.right_recv_halo.get() == 14))
    assert(np.all(da.bottom_recv_halo.get() == 10))
    assert(np.all(da.top_recv_halo.get() == 16))
    assert(np.all(da.front_recv_halo.get() == 4))
    assert(np.all(da.back_recv_halo.get() == 22))

MPI.Finalize()
