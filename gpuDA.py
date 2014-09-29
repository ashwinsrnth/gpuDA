from mpi4py import MPI
import numpy as np
from pycuda import autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

class GpuDA:

    def __init__(self, comm, local_dims, proc_sizes, stencil_width):
        self.comm = comm
        self.local_dims = local_dims
        self.proc_sizes = proc_sizes
        self.stencil_width = stencil_width
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        assert(isinstance(comm, MPI.Cartcomm))
        assert(self.size == reduce(lambda a,b: a*b, proc_sizes))

        self._create_halo_arrays()
    
    def halo_swap(self, array):
        npz, npy, npx = self.proc_sizes
        nz, ny, nx = self.local_dims
        zloc, yloc, xloc = self.comm.Get_topo()[2]
        sw = self.stencil_width

        # copy from arrays to send halos:
        self._copy_array_to_halo(array, self.left_send_halo, [sw, ny, nz], [0, 0, 0])
        self._copy_array_to_halo(array, self.right_send_halo, [sw, ny, nz], [nx-1, 0, 0])

        # perform swaps in x-direction
        sendbuf = [self.right_send_halo.gpudata.as_buffer(self.right_send_halo.nbytes), MPI.DOUBLE]
        recvbuf = [self.left_recv_halo.gpudata.as_buffer(self.left_recv_halo.nbytes), MPI.DOUBLE]
        self._forward_swap(sendbuf, recvbuf, self.rank-1, self.rank+1, xloc, npx)

        sendbuf = [self.left_send_halo.gpudata.as_buffer(self.left_send_halo.nbytes), MPI.DOUBLE]
        recvbuf = [self.right_recv_halo.gpudata.as_buffer(self.right_recv_halo.nbytes), MPI.DOUBLE]
        self._backward_swap(sendbuf, recvbuf, self.rank+1, self.rank-1, xloc, npx)

        # perform swaps in y-direction:
        #sendbuf = [a_gpu.gpudata.as_buffer(a_gpu.nbytes), 1, self.front]
        #recvbuf = [b_gpu.gpudata.as_buffer(b_gpu.nbytes), 1, self.back]
        #self._forward_swap(sendbuf, recvbuf, self.rank-npx, self.rank+npx, yloc, npy)
        
        #sendbuf = [a_gpu.gpudata.as_buffer(a_gpu.nbytes), 1, self.back]
        #recvbuf = [b_gpu.gpudata.as_buffer(b_gpu.nbytes), 1, self.front]
        #self._backward_swap(sendbuf, recvbuf, self.rank+npx, self.rank-npx, yloc, npy)

        # perform swaps in z-direction
        #sendbuf = [a_gpu.gpudata.as_buffer(a_gpu.nbytes), 1, self.top]
        #recvbuf = [b_gpu.gpudata.as_buffer(b_gpu.nbytes), 1, self.bottom]
        #self._forward_swap(sendbuf, recvbuf, self.rank-npx*npy, self.rank+npx*npy, zloc, npz)
        
        #sendbuf = [a_gpu.gpudata.as_buffer(a_gpu.nbytes), 1, self.bottom]
        #recvbuf = [b_gpu.gpudata.as_buffer(b_gpu.nbytes), 1, self.top]
        #self._backward_swap(sendbuf, recvbuf, self.rank+npx*npy, self.rank-npx*npy, zloc, npz)
        
    def _forward_swap(self, sendbuf, recvbuf, src, dest, loc, dimprocs):
        if loc > 0 and loc < dimprocs-1:
            self.comm.Sendrecv(sendbuf=sendbuf, dest=dest, sendtag=10, recvbuf=recvbuf, recvtag=10, source=src)
          
        elif loc == 0 and dimprocs > 1:
            self.comm.Send(sendbuf, dest=dest, tag=10)

        elif loc == dimprocs-1 and dimprocs > 1:
            self.comm.Recv(recvbuf, source=src, tag=10)

    def _backward_swap(self, sendbuf, recvbuf, src, dest, loc, dimprocs):
        if loc > 0 and loc < dimprocs-1:
            self.comm.Sendrecv(sendbuf=sendbuf, dest=dest, sendtag=10, recvbuf=recvbuf, recvtag=10, source=src)

        elif loc == 0 and dimprocs > 1:
            self.comm.Recv(recvbuf, source=src, tag=10)

        elif loc == dimprocs-1 and dimprocs > 1:
            self.comm.Send(sendbuf, dest=dest, tag=10)

    def _create_halo_arrays(self):
        nz, ny, nx = self.local_dims
        sw = self.stencil_width
        # create two halo regions for each face, one holding
        # the halo values to send, and the other holding
        # the halo values to receive.

        self.left_recv_halo = gpuarray.empty([nz,ny,sw], dtype=np.float64)
        self.left_send_halo = self.left_recv_halo.copy()
        self.right_recv_halo = self.left_recv_halo.copy()
        self.right_send_halo = self.left_recv_halo.copy()
    
    def _copy_array_to_halo(self, array, halo, copy_dims, copy_offsets, dtype=np.float64):
        # copy from 3-d array to 2-d halo
        #
        # Paramters:
        # array, halo:  gpuarrays involved in the copy.
        # copy_dims: number of elements to copy in (x, y, z) directions
        # copy_offsets: offsets at the source in (x, y, z) directions
        
        w, h, d = copy_dims
        x_offs, y_offs, z_offs = copy_offsets
        
        # TODO: a general type size
        #type_size = dtype.itemsize

        copier = cuda.Memcpy3D()
        copier.set_src_device(array.gpudata)
        copier.set_dst_device(halo.gpudata)

        copier.src_x_in_bytes = x_offs*8
        copier.src_y = y_offs

        copier.src_pitch = array.strides[1]
        copier.dst_pitch = halo.strides[1]
        copier.src_height = h
        copier.dst_height = h

        copier.width_in_bytes = w*8
        copier.height = h
        copier.depth = d

        # perform the copy:
        copier()
        
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    npx = 3
    npy = 1
    npz = 1
        
    assert(npx*npy*npz == size)

    nx = 3
    ny = 512
    nz = 512

    a = np.arange(nx*ny*nz, dtype=np.float64).reshape([nz, ny, nx])
    a_gpu = gpuarray.to_gpu(a)
    
    comm = comm.Create_cart([npz, npy, npx], reorder=True)
    da = GpuDA(comm, [nz, ny, nx], [npz, npy, npx], 1)
    da.halo_swap(a_gpu)
    MPI.Finalize()

