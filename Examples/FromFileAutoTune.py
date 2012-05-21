#Copyright (c) 2012, Eng Eder de Souza
#AutoTune from Wav File Example!

import sys
import numpy
import scikits.audiolab as audiolab 
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
CHUNK=2048


NewSignal=[]

if len(sys.argv)<3 :
        print 'Usage: %s <Input audio file.wav> <Output audio file.wav>' %sys.argv[0]
        sys.exit(0)

IN=sys.argv[1]
OUT=sys.argv[2]





f = audiolab.Sndfile(IN, 'r')

FS = f.samplerate
nchannels  = f.channels



datas = f.read_frames(CHUNK, dtype=numpy.float32)



while datas !='':
	
	print "."
	Signal = datas.data[:]
	rawfromC=AutoTune.Tuner(Signal,FS,CHUNK,SCALE_ROTATE,SCALE_ROTATE,LFO_QUANT,CONCERT_A,FIXED_PITCH,FIXED_PULL,CORR_STR,CORR_SMOOTH,PITCH_SHIFT,LFO_DEPTH,LFO_RATE,LFO_SHAPE,LFO_SYMM,FORM_WARP,MIX,KEY)
	
	for s in rawfromC:
		NewSignal.append(s)

	try:
		datas = f.read_frames(CHUNK, dtype=numpy.float32)
	
	except:
		break
	
	
	
	


array = numpy.array(NewSignal)

fmt         = audiolab.Format('wav', 'pcm32')



# making the file .wav
afile =  audiolab.Sndfile(OUT, 'w', fmt, nchannels, FS)



#writing in the file
afile.write_frames(array)

print "Done!"


