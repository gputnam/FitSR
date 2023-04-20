from setuptools import setup, Extension
import numpy

setup(
  name="fitSR_chelpers",
  ext_modules=[Extension("fitSR_chelpers", ["src/fitSR_chelpers.C"])],
  include_dirs=[numpy.get_include()],
  version="0.1",
  description="Helper functions for Signal Response fitting",
  author="Gray Putnam",
  # packages=["fitSR_chelpers"],
  install_requires=["numpy"],
  zip_safe=False
)
