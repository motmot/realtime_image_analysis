import os
from setuptools import setup, Extension

from motmot_utils import get_svnversion_persistent
version_str = '0.4.dev%(svnversion)s'
version = get_svnversion_persistent('realtime_image_analysis/version.py',version_str)

install_requires = ['FastImage']

import FastImage_util
vals = FastImage_util.get_build_info(ipp_static=False,
                                     ipp_version='5.1')

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
      version=version,
      author="Andrew Straw",
      author_email="strawman@astraw.com",
      description="several image analysis functions that require Intel IPP and FastImage",
      packages = ['realtime_image_analysis'],
      ext_modules= ext_modules,
      install_requires = install_requires,
      zip_safe = True,
      )
