from distutils.core import setup, Extension
 
Tuner = Extension('AutoTune', ['PyAutoTuner.c', 'autotalent.c', 'mayer_fft.c'])
 

setup (name = 'PackageName',
        version = '0.1b',
        author = "Eng Eder de Souza",
        author_email = "ederwander@gmail.com",
        url = "http://github.com/ederwander/PyAutoTune",
        description = 'AutoTune package',
        ext_modules = [Tuner])