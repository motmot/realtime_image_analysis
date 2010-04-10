import os
from setuptools import setup, Extension, find_packages
import pkg_resources # make sure FastImage is importable

import motmot.FastImage.FastImage as FastImage

ext_modules = []

if 1:
    # Pyrex build of realtime_image_analysis
    realtime_image_analysis_sources=['src/realtime_image_analysis.c', # compile with Cython
                                     'src/c_fit_params.c',
                                     'src/eigen.c',
                                     'src/c_time_time.c',
                                     'src/fic.c',
                                     ]
    ext_modules.append(Extension(name='motmot.realtime_image_analysis.realtime_image_analysis',
                                 sources=realtime_image_analysis_sources,
                                 libraries=['fwBase','fwImage','cv'],
                                 ))

setup(name='motmot.realtime_image_analysis',
      description="several image analysis functions that require Intel IPP and FastImage",
      long_description=
"""This code serves as the basis for at least 2 different classes of
realtime trackers: 2D only trackers with no consideration of camera
calibration and potentially-3D trackers with camera calibration and
distortion information.""",
      version='0.5.7',
      author="Andrew Straw",
      author_email="strawman@astraw.com",
      url='http://code.astraw.com/projects/motmot',
      license='BSD',
      namespace_packages = ['motmot'],
      packages = find_packages(),
      ext_modules= ext_modules,
      )
