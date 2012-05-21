/* Copyright 2012, Eng Eder de Souza 

   This program is free software; you can redistribute it and/or modify        
   it under the terms of the GNU General Public License as published by        
   the Free Software Foundation; either version 2 of the License, or           
   (at your option) any later version.                                         
                                                                                
   This program is distributed in the hope that it will be useful,             
   but WITHOUT ANY WARRANTY; without even the implied warranty of              
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               
   GNU General Public License for more details.                                
                                                                                
   You should have received a copy of the GNU General Public License           
   along with this program; if not, write to the Free Software                 
   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.  

*/


#include <Python.h>
#include <numpy/arrayobject.h>
#include "mayer_fft.h"
#include "autotalent.h"


static PyObject *Tuner(PyObject *self, PyObject *args)

{

	float *signal, *buffer, concert_a, fixed_pitch, fixed_pull, corr_str, corr_smooth, pitch_shift, lfo_depth, lfo_rate, lfo_shape, lfo_symm, form_warp, mix;

	char *key;

	int fs, FrameSize, zz, i, scale_rotate, lfo_quant, form_corr;

	PyObject *obj;
	PyArrayObject *arr;

	if (!PyArg_ParseTuple(args, "s#iiiiiffffffffffffc", &signal,&zz,&fs,&FrameSize,&scale_rotate,&lfo_quant,&form_corr,&concert_a,&fixed_pitch,&fixed_pull,&corr_str,&corr_smooth,&pitch_shift,&lfo_depth,&lfo_rate,&lfo_shape,&lfo_symm,&form_warp,&mix,&key))
		return NULL;

	Py_BEGIN_ALLOW_THREADS;

	instantiateAutotalentInstance(fs);

	buffer = (float*)malloc(FrameSize*sizeof(float));

	if (!buffer) {
    		printf("\nError: No memory\n");
    		exit(1);
  	}


	initializeAutotalent(&concert_a, &key, &fixed_pitch, &fixed_pull, &corr_str, &corr_smooth, &pitch_shift, &scale_rotate, &lfo_depth, &lfo_rate, &lfo_shape, &lfo_symm, &lfo_quant, &form_corr, &form_warp, &mix);

	for(i=0; i<FrameSize; i++) 	buffer[i] = *(signal++);

	processSamples(buffer, FrameSize);

	arr = (PyArrayObject *)PyArray_SimpleNewFromData(1, &FrameSize,PyArray_FLOAT,(char *) buffer);

	Py_END_ALLOW_THREADS;

	return PyArray_Return(arr);

	free(buffer);

	*buffer = 0;

	freeAutotalentInstance();


}

static PyObject *ErrorObject;

static PyMethodDef TuneMethod[] = 
{
    {"Tuner", Tuner, METH_VARARGS, "AutoTune Signal"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC 
initAutoTune(void)
{
	(void) Py_InitModule("AutoTune", TuneMethod);
  	import_array();
  	ErrorObject = PyString_FromString("AutoTune.error");
 	if (PyErr_Occurred())
 		Py_FatalError("can't initialize module AutoTune");
}
