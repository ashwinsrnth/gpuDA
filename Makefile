MPIFLAGS=--mca btl_openib_warn_nonexistent_if 0 --mca btl_openib_want_cuda_gdr 1 --mca pml ob1

time:
	mpiexec -n 27 ${MPIFLAGS} python time_gpuDA.py
	rm *.pyc
