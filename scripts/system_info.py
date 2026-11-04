"""Print environment details alongside reproducible benchmark data."""
import os
import platform
import shutil
import subprocess
import sys

print("Python:",sys.version.replace("\n"," "))
print("Operating system:",platform.platform())
print("Processor:",platform.processor() or platform.machine())
print("CPU logical count:",os.cpu_count())
for name in ("numpy","numba","matplotlib","mpi4py","streamlit"):
    try:
        m=__import__(name)
        print(name+":",getattr(m,"__version__","installed"))
        if name=="mpi4py":
            from mpi4py import MPI
            print("MPI vendor:",MPI.get_vendor())
    except ImportError:
        print(name+":","not installed")
for name in ("mpirun","mpiexec"):
    if shutil.which(name):print(name+":",shutil.which(name))
print("NBODY_KERNEL:",os.environ.get("NBODY_KERNEL","numpy (default)"))
