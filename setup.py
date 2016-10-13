from setuptools import setup

setup(name='gpuda',
        version=0.1,
        description='distributed arrays for GPUs',
        url='http://github.com/shwina/gpuda',
        author='Ashwin Srinath',
        author_email='atrikut@clemson.edu',
        license='MIT',
        py_modules=['gpuda'],
        setup_requires=['mpi4py'],
        dependency_links=['https://mathema.tician.de/software/pycuda/'],
        zip_safe=False)
