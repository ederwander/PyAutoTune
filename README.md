PyAutoTune
==========

Copyright 2012-2020, Eng Eder de Souza

This module provide one Tuner function to AutoTune monophonic sound chunks in Float 32bits. Designed for autotune in
realtime microphone or recorded file.

PyAutoTune is one port from the source written by Tom Baran http://tombaran.info/autotalent.html.

This module provide an excellent performance in real time, tested in Linux and Windows!

==REQUIREMENTS==
==========

Python 2.6 or later

One C compiler 

NumPy 1.0 or later

For Python <= 2.7 use the [ForPy2X](https://github.com/ederwander/PyAutoTune/tree/master/ForPy2X) codes to install

==WINDOWS INSTALLATION==
==========

PyAutoTune is packaged as Python and C source using distutils.  To install:

> pip install setuptools numpy pyaudio

- Setuptools - used for compiler

- Numpy - used for audio arrays

- pyaudio - play and test code examples 

Download and install one compliler, for example you can choose mingw 32 or 64 bits in windows SO:

https://sourceforge.net/projects/mingw/files/latest/download

https://sourceforge.net/projects/mingw-w64/

Create one file distutils.cfg in your python distutils PATH instalation Ex:C:\Python37\Lib\distutils
```
[build]
compiler=mingw32


 [build_ext]
include_dirs= C:\Python37\Lib\site-packages\numpy\core\include
```




Set the PATH to you mingw and python, ex in CMD.

> set PATH=C:\MinGW\bin;C:\Python37;%PATH%

if you installed mingw 64bits point the path correctly

Now you're ready to compile:

> python.exe setup.py install


==RUNNING IN DOCKER==
==========
Build docker image: (you can replace `autotune` with any name)
```
docker build -t autotune .
```

Run a container using created image:  
`--rm` removes container when it stopped  
`-v` flag can be used to map `Examples` folder on host filesystem to `Examples` folder in the container. You can add audio files on host machine and edit autotune parameters in `Examples/TuneAndSaveToFile.py` in editor. Output file will also available on host filesystem.

```
docker run --rm -v $PWD/Examples:/app/Examples -it autotune /bin/sh
```

Run example:

```
cd Examples
python TuneAndSaveToFile.py teste.wav out.wav
```

==EXAMPLES==
==========

The package is imported with 'import AutoTune'.

You can find two simple example (real-time and from-file) in the folder [examples](http://github.com/ederwander/PyAutoTune/tree/master/Examples)!


