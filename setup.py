import sys
import os.path
import numpy
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

import pkg_resources # make sure FastImage is importable
import motmot.FastImage as fi_mod
import motmot.FastImage.FastImage as FastImage
import motmot.FastImage.util as FastImage_util

# build with same IPP as FastImage
ipp_root = os.environ['IPPROOT']
vals = FastImage_util.get_build_info(ipp_arch=FastImage.get_IPP_arch(),
                                     ipp_static=FastImage.get_IPP_static(),
                                     ipp_root=ipp_root)

ext_modules = []

if 1:
    realtime_image_analysis_sources=['motmot/realtime_image_analysis/realtime_image_analysis.pyx',
                                     'src/c_fit_params.cpp',
                                     'src/eigen.c',
                                     'src/c_time_time.c',
                                     ]
    ext_modules.append(Extension(name='motmot.realtime_image_analysis.realtime_image_analysis',
                                 sources=realtime_image_analysis_sources,
                                 include_dirs=vals['ipp_include_dirs']+['src']+[numpy.get_include(), fi_mod.get_include()],
                                 library_dirs=vals['ipp_library_dirs'],
                                 libraries=vals['ipp_libraries'],
                                 define_macros=vals['ipp_define_macros'],
                                 extra_objects=vals['ipp_extra_objects'],
                                 language="c++",
                                 ))
    ext_modules = cythonize(ext_modules)

setup(name='motmot.realtime_image_analysis',
      description="several image analysis functions that require Intel IPP and FastImage",
      long_description=
"""This code serves as the basis for at least 2 different classes of
realtime trackers: 2D only trackers with no consideration of camera
calibration and potentially-3D trackers with camera calibration and
distortion information.""",
      version='0.8.0', # also in motmot/realtime_image_analysis/__init__.py
      author="Andrew Straw",
      author_email="strawman@astraw.com",
      url='http://code.astraw.com/projects/motmot',
      license='BSD',
      namespace_packages = ['motmot'],
      packages = find_packages(),
      include_package_data=True,
      zip_safe= False,
      ext_modules= ext_modules,
      )
