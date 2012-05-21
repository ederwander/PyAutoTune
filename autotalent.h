#ifndef AUTOTALENT_H
#define AUTOTALENT_H

void instantiateAutotalentInstance(unsigned long sampleRate);

void initializeAutotalent(float* concertA, char* key, float* fixedPitch, float* fixedPull,
                          float* correctStrength, float* correctSmooth, float* pitchShift, int* scaleRotate,
                          float* lfoDepth, float* lfoRate, float* lfoShape, float* lfoSym, int* lfoQuant,
                          int* formCorr, float* formWarp, float* mix);

void processSamples(float* samples, int sampleSize);

void freeAutotalentInstance();

#endif
