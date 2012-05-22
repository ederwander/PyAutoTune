#Copyright (c) 2012, Eng Eder de Souza
#AutoTune Real-time from Microphone Example!


from struct import pack
import numpy
import pyaudio
import AutoTune

FORM_CORR=0
SCALE_ROTATE=0
LFO_QUANT=0
CONCERT_A=440.0
FIXED_PITCH=2.0
FIXED_PULL=0.1
CORR_STR=1.0
CORR_SMOOTH=0.0
PITCH_SHIFT=1.0
LFO_DEPTH=0.1
LFO_RATE=1.0
LFO_SHAPE=0.0
LFO_SYMM=0.0
FORM_WARP=0.0
MIX=1.0
KEY="c"
FS=44100
CHUNK=2048


# Initialize PyAudio
pyaud = pyaudio.PyAudio()

# Open input stream, 32-bit mono at 44100 Hz
stream = pyaud.open(
    format = pyaudio.paFloat32,
    channels = 1,
    rate = FS,
    output = True,
    input = True)



#while True:
for i in range(0, FS / CHUNK * 10):

		rawsamps = stream.read(CHUNK)
		samps = numpy.fromstring(rawsamps, 'Float32');
		Signal = samps.data[:]
		rawfromC=AutoTune.Tuner(Signal,FS,CHUNK,SCALE_ROTATE,SCALE_ROTATE,LFO_QUANT,CONCERT_A,FIXED_PITCH,FIXED_PULL,CORR_STR,CORR_SMOOTH,PITCH_SHIFT,LFO_DEPTH,LFO_RATE,LFO_SHAPE,LFO_SYMM,FORM_WARP,MIX,KEY)
		uaaaaaa2 = pack("%df"%(len(rawfromC)), *list(rawfromC))
		stream.write(uaaaaaa2, CHUNK)
