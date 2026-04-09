from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize("*.pyx", language_level=3),
    include_dirs=[numpy.get_include()],
)

# To compile the Cython modules, run this command in the terminal:
# python setup.py build_ext --inplace
