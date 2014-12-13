import sys
import os.path
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

import pkg_resources # make sure FastImage is importable
import motmot.FastImage.FastImage as FastImage
import motmot.FastImage.util as FastImage_util

# build with same IPP as FastImage
major,minor,build = FastImage.get_IPP_version()
vals = FastImage_util.get_build_info(ipp_static=FastImage.get_IPP_static(),
                                     ipp_version='%d.%d'%(major,minor),
                                     ipp_arch=FastImage.get_IPP_arch(),
                                     )

ipp_sources = vals.get('ipp_sources',[])
ipp_include_dirs = vals.get('ipp_include_dirs',[])
ipp_library_dirs = vals.get('ipp_library_dirs',[])
ipp_libraries = vals.get('ipp_libraries',[])
ipp_define_macros = vals.get('ipp_define_macros',[])
ipp_extra_link_args = vals.get('extra_link_args',[])
ipp_extra_compile_args = vals.get('extra_compile_args',[])

if os.path.exists('/usr/include/opencv-2.3.1'):
    #ROS packaging
    cv_inc_dir = 'opencv-2.3.1'
else:
    #debian packaging
    cv_inc_dir = ''

cv_include_dirs = ['/usr/include/'+cv_inc_dir]
cv_libraries = ['opencv_core',
                'opencv_legacy',
                'opencv_imgproc',
                ]

ext_modules = []

if 1:
    realtime_image_analysis_sources=['src/realtime_image_analysis.pyx',
                                     'src/c_fit_params.cpp',
                                     'src/fic.c',
                                     'src/eigen.c',
                                     'src/c_time_time.c',
                                     ]+ipp_sources
    ext_modules.append(Extension(name='motmot.realtime_image_analysis.realtime_image_analysis',
                                 sources=realtime_image_analysis_sources,
                                 include_dirs=ipp_include_dirs+cv_include_dirs,
                                 library_dirs=ipp_library_dirs,
                                 libraries=ipp_libraries+cv_libraries+['stdc++'],
                                 define_macros=ipp_define_macros,
                                 extra_link_args=ipp_extra_link_args,
                                 extra_compile_args=ipp_extra_compile_args,
                                 language="c++",
                                 ))
    ext_modules = cythonize(ext_modules)

setup(name='motmot.realtime_image_analysis-ipp',
      description="several image analysis functions that require Intel IPP and FastImage",
      long_description=
"""This code serves as the basis for at least 2 different classes of
realtime trackers: 2D only trackers with no consideration of camera
calibration and potentially-3D trackers with camera calibration and
distortion information.""",
      version='0.5.9',
      author="Andrew Straw",
      author_email="strawman@astraw.com",
      url='http://code.astraw.com/projects/motmot',
      license='BSD',
      namespace_packages = ['motmot'],
      packages = find_packages(),
      ext_modules= ext_modules,
      )
