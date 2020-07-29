from distutils.core import setup, Extension
import numpy
 
Tuner = Extension('AutoTune', ['PyAutoTuner.c', 'autotalent.c', 'mayer_fft.c'])
 

setup (name = 'PyAutoTune',
        version = '0.1b',
        author = "Eng Eder de Souza",
        author_email = "ederwander@gmail.com",
        url = "http://github.com/ederwander/PyAutoTune",
        description = 'AutoTune package',
        ext_modules = [Tuner],
        include_dirs = [numpy.get_include()])
