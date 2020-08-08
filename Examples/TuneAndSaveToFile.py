from struct import pack
from wave import open
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


if len(sys.argv)<3 :
        print("Usage: %s <Input audio file.wav> <Output audio file.wav>" %sys.argv[0])
        sys.exit(0)

IN=sys.argv[1]
OUT=sys.argv[2]



wf = open(IN, 'rb')

# If Stereo
if wf.getnchannels() == 2:
    print("Just mono files")
    sys.exit(0)



signal = wf.readframes(-1)
FS = wf.getframerate()
scale = 1<<15
intsignal = np.frombuffer(signal, dtype=np.int16)
floatsignal = np.float32(intsignal) / scale


####Setup to Write an Out Wav File####

fout=open(OUT,'w')
fout.setnchannels(1) # Mono
fout.setsampwidth(2) # Sample is 2 Bytes (2) if int16 = short int
fout.setframerate(FS)# Sampling Frequency
fout.setcomptype('NONE','Not Compressed')

for i in range(0, len(floatsignal), CHUNK):
        
        SignalChunk = (floatsignal[i:i+CHUNK])
        if i+CHUNK > len(floatsignal):
            CHUNK=len(SignalChunk)
        rawfromC = AutoTune.Tuner(SignalChunk,FS,CHUNK,SCALE_ROTATE,LFO_QUANT,FORM_CORR,CONCERT_A,FIXED_PITCH,FIXED_PULL,CORR_STR,CORR_SMOOTH,PITCH_SHIFT,LFO_DEPTH,LFO_RATE,LFO_SHAPE,LFO_SYMM,FORM_WARP,MIX,KEY)
        shortIntvalues  = np.int16(np.asarray(rawfromC)*(scale))
        outdata = pack("%dh"%len(shortIntvalues), *(shortIntvalues))
        fout.writeframesraw(outdata)
            
#close write wav
fout.close()

    
