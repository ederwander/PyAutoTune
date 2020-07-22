from struct import pack
from wave import open
import pyaudio
import sys
import numpy as np
import AutoTune

FORM_CORR=0
SCALE_ROTATE=0
LFO_QUANT=0
CONCERT_A=440.0
FIXED_PITCH=2.0
FIXED_PULL=0.1
CORR_STR=1.0
CORR_SMOOTH=0.0
PITCH_SHIFT=0.0
LFO_DEPTH=0.1
LFO_RATE=1.0
LFO_SHAPE=0.0
LFO_SYMM=0.0
FORM_WARP=0.0
MIX=1.0
KEY="c".encode()
CHUNK=4096


wf = open('teste.wav', 'rb')

# If Stereo
if wf.getnchannels() == 2:
    print("Just mono files")
    sys.exit(0)


# Initialize PyAudio
pyaud = pyaudio.PyAudio()


# Open stream
stream = pyaud.open(format =  pyaudio.paFloat32,
               channels = wf.getnchannels(),
               rate = wf.getframerate(),
               output = True)




signal = wf.readframes(-1)
FS = wf.getframerate()
intsignal = np.frombuffer(signal, dtype=np.int16)
floatsignal  = np.float32(intsignal) / 2**15


for i in range(0, len(floatsignal), CHUNK):
        
        SignalChunk = (floatsignal[i:i+CHUNK])
        if i+CHUNK > len(floatsignal):
            CHUNK=len(SignalChunk)
        rawfromC=AutoTune.Tuner(SignalChunk,FS,CHUNK,SCALE_ROTATE,LFO_QUANT,FORM_CORR,CONCERT_A,FIXED_PITCH,FIXED_PULL,CORR_STR,CORR_SMOOTH,PITCH_SHIFT,LFO_DEPTH,LFO_RATE,LFO_SHAPE,LFO_SYMM,FORM_WARP,MIX,KEY)
        out = pack("%df"%len(rawfromC), *(rawfromC))
        stream.write(out)



    
# stop stream (4)
stream.stop_stream()
stream.close()

# close PyAudio (5)
pyaud.terminate()

