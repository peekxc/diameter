import os
import sysconfig
from setuptools import setup, Extension
#from setuptools_cpp import ExtensionBuilder, Pybind11Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
base_path = os.path.dirname(__file__)

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-std=c++17", "-Wall", "-Wextra", "-fopenmp=libomp", "-O3"]

import distutils.sysconfig
flags = distutils.sysconfig.get_config_var("CFLAGS")
print(f"COMPILER FLAGS: { str(flags) }")


ext_modules = [
  Pybind11Extension(
    'diameter_ext', 
    sources = ['src/diameter/extensions/diameter.cpp'], 
    include_dirs=['/Users/mpiekenbrock/diameter/extern/pybind11/include'], 
    extra_compile_args=extra_compile_args,
    language='c++17', 
    cxx_std=1
  )
  # Extension(
  #   'diameter_ext',
  #   sources = ['src/diameter/extensions/diameter.cpp'],
  #   include_dirs=['/Users/mpiekenbrock/diameter/extern/pybind11/include', os.path.join(base_path, 'include')],
  #   language='c++11', 
  #   extra_compile_args=extra_compile_args
  # )
]

setup(
  name="diameter",
  version="0.1.0",
  package_dir={"":"src"},
  ext_modules=ext_modules,
  cmdclass={"build_ext": build_ext},
  zip_safe=False
)
# For python develop: pip install --editable .
# For c++ develop: python3 -m build --no-isolation --wheel --skip-dependency-check