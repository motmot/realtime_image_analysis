import os
from setuptools import setup, Extension

install_requires = ['FastImage']

import FastImage
major,minor,build = FastImage.get_IPP_version()
import FastImage_util

# build with same IPP as FastImage
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

ext_modules = []

if 1:
    # Pyrex build of realtime_image_analysis
    realtime_image_analysis_extension_name='_realtime_image_analysis'
    realtime_image_analysis_sources=['src/_realtime_image_analysis.pyx',
                                     'src/c_fit_params.c',
                                     'src/eigen.c',
                                     'src/c_time_time.c',
                                     ]+ipp_sources
    ext_modules.append(Extension(name=realtime_image_analysis_extension_name,
                                 sources=realtime_image_analysis_sources,
                                 include_dirs=ipp_include_dirs,
                                 library_dirs=ipp_library_dirs,
                                 libraries=ipp_libraries,
                                 define_macros=ipp_define_macros,
                                 extra_link_args=ipp_extra_link_args,
                                 extra_compile_args=ipp_extra_compile_args,
                                 ))
    
if os.name.startswith('posix'):
    install_requires.append('posix_sched')

setup(name='realtime_image_analysis',
      description="several image analysis functions that require Intel IPP and FastImage",
      long_description=
"""This code serves as the basis for at least 2 different classes of
realtime trackers: 2D only trackers with no consideration of camera
calibration and potentially-3D trackers with camera calibration and
distortion information.""",
      version='0.5.3',
      author="Andrew Straw",
      author_email="strawman@astraw.com",
      url='http://code.astraw.com/projects/motmot',
      license='BSD',
      packages = ['realtime_image_analysis'],
      ext_modules= ext_modules,
      install_requires = install_requires,
      zip_safe = True,
      )
