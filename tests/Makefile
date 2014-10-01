MPIFLAGS=--mca btl_openib_warn_nonexistent_if 0 --mca btl_openib_want_cuda_gdr 1 --mca pml ob1

test_1d:
	@echo "testing 1d case with 3 procs: "
	mpiexec -n 3 ${MPIFLAGS} python test_1d.py

test_3d:
	@echo "testing 3d case with 27 procs: "
	mpiexec -n 27 ${MPIFLAGS} python test_3d.py

